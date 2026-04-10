#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter and volume confirmation
# - Long when price breaks above Donchian upper band (20-period high) AND 1d close > 1d open (bullish daily candle) AND volume > 2.0x 20-period volume SMA
# - Short when price breaks below Donchian lower band (20-period low) AND 1d close < 1d open (bearish daily candle) AND volume > 2.0x 20-period volume SMA
# - Exit: ATR-based trailing stop (3x ATR) or Donchian middle band reversion
# - Uses 1d for signal direction (trend bias) and 4h for precise entry timing
# - Session filter: 08-20 UTC to avoid low-volume Asian session noise
# - Position sizing: 0.25 discrete level to control drawdown and minimize fee churn
# - Target: 25-50 trades/year (100-200 total over 4 years) to minimize fee drag while maintaining statistical significance

name = "4h_1d_donchian_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return signals
    
    # Calculate 1d candle direction (bullish/bearish) for trend filter
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    # Bullish 1d candle: close > open
    bullish_1d = close_1d > open_1d
    bearish_1d = close_1d < open_1d
    # Align to 4h timeframe with proper delay (completed 1d bar only)
    bullish_1d_aligned = align_htf_to_ltf(prices, df_1d, bullish_1d)
    bearish_1d_aligned = align_htf_to_ltf(prices, df_1d, bearish_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h Donchian channels (20-period)
    # Upper band = highest high over 20 periods
    # Lower band = lowest low over 20 periods
    # Middle band = (upper + lower) / 2
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Calculate ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period volume SMA for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Track highest high since entry for trailing stop (long)
    # Track lowest low since entry for trailing stop (short)
    highest_since_entry = np.full(n, np.nan)
    lowest_since_entry = np.full(n, np.nan)
    
    for i in range(20, n):  # Start from 20 to have sufficient lookback
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_middle[i]) or
            np.isnan(atr[i]) or np.isnan(volume_sma_20[i]) or
            np.isnan(bullish_1d_aligned[i]) or np.isnan(bearish_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 4h volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > 2.0 * volume_sma_20[i]
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_upper[i-1]  # Break above upper band (using previous close to avoid look-ahead)
        breakout_down = close[i] < donchian_lower[i-1]  # Break below lower band
        
        if position == 0:  # Flat - look for entry
            # Long: price breaks above upper band AND 1d bullish AND volume confirmation
            if breakout_up and bullish_1d_aligned[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
                highest_since_entry[i] = high[i]  # Initialize trailing stop
            # Short: price breaks below lower band AND 1d bearish AND volume confirmation
            elif breakout_down and bearish_1d_aligned[i] and vol_confirm:
                position = -1
                signals[i] = -0.25
                lowest_since_entry[i] = low[i]  # Initialize trailing stop
            else:
                signals[i] = 0.0
                # Carry forward NaN values for tracking
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
        elif position == 1:  # Long position - look for exit
            # Update highest high since entry
            highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            
            # ATR trailing stop: exit if price drops 3*ATR below highest high since entry
            trailing_stop = highest_since_entry[i] - 3.0 * atr[i]
            
            # Exit conditions: trailing stop hit OR reversion to middle band
            exit_condition = (close[i] < trailing_stop) or (close[i] < donchian_middle[i])
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
                # Reset tracking arrays
                highest_since_entry[i] = np.nan
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = 0.25
                # Propagate tracking values
                highest_since_entry[i] = highest_since_entry[i]
                lowest_since_entry[i] = lowest_since_entry[i-1]
        else:  # position == -1 (Short position) - look for exit
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            
            # ATR trailing stop: exit if price rises 3*ATR above lowest low since entry
            trailing_stop = lowest_since_entry[i] + 3.0 * atr[i]
            
            # Exit conditions: trailing stop hit OR reversion to middle band
            exit_condition = (close[i] > trailing_stop) or (close[i] > donchian_middle[i])
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
                # Reset tracking arrays
                highest_since_entry[i] = np.nan
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = -0.25
                # Propagate tracking values
                highest_since_entry[i] = highest_since_entry[i-1]
                lowest_since_entry[i] = lowest_since_entry[i]
    
    return signals