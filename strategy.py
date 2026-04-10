#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
# - Long when price breaks above 20-day high AND 1w close > 1w open (bullish weekly candle) AND volume > 1.5x 20-day volume SMA
# - Short when price breaks below 20-day low AND 1w close < 1w open (bearish weekly candle) AND volume > 1.5x 20-day volume SMA
# - Exit: ATR trailing stop (2.0x ATR) or reversion to 20-day midpoint
# - Uses 1w for signal direction (trend bias) and 1d for precise entry timing
# - Position sizing: 0.25 discrete level to control drawdown and minimize fee churn
# - Target: 10-25 trades/year (40-100 total over 4 years) to minimize fee drag while maintaining statistical significance
# - Donchian channels work in both bull (breakouts) and bear (breakdowns) markets with proper trend filter
# - Uses previous completed 1w bar for trend filter to avoid look-ahead

name = "1d_donchian_20_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1w data ONCE before loop (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return signals
    
    # Calculate 1w candle direction (bullish/bearish) for trend filter
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    # Bullish 1w candle: close > open
    bullish_1w = close_1w > open_1w
    bearish_1w = close_1w < open_1w
    # Align to 1d timeframe with proper delay (completed 1w bar only)
    bullish_1w_aligned = align_htf_to_ltf(prices, df_1w, bullish_1w)
    bearish_1w_aligned = align_htf_to_ltf(prices, df_1w, bearish_1w)
    
    # Calculate ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-day volume SMA for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 20-day Donchian channels
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max_20 + low_min_20) / 2.0
    
    # Track highest high since entry for trailing stop (long)
    # Track lowest low since entry for trailing stop (short)
    highest_since_entry = np.full(n, np.nan)
    lowest_since_entry = np.full(n, np.nan)
    
    for i in range(20, n):  # Start from 20 to have sufficient lookback
        # Skip if any required data is invalid
        if (np.isnan(atr[i]) or np.isnan(volume_sma_20[i]) or
            np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or
            np.isnan(bullish_1w_aligned[i]) or np.isnan(bearish_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.5x 20-day volume SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        # Donchian breakout signals
        breakout_up = close[i] > high_max_20[i-1]  # Break above 20-day high
        breakout_down = close[i] < low_min_20[i-1]  # Break below 20-day low
        
        if position == 0:  # Flat - look for entry
            # Long: price breaks above 20-day high AND 1w bullish AND volume confirmation
            if breakout_up and bullish_1w_aligned[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
                highest_since_entry[i] = high[i]  # Initialize trailing stop
            # Short: price breaks below 20-day low AND 1w bearish AND volume confirmation
            elif breakout_down and bearish_1w_aligned[i] and vol_confirm:
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
            
            # ATR trailing stop: exit if price drops 2.0*ATR below highest high since entry
            trailing_stop = highest_since_entry[i] - 2.0 * atr[i]
            
            # Exit conditions: trailing stop hit OR reversion to midpoint
            exit_condition = (close[i] < trailing_stop) or (close[i] < donchian_mid[i])
            
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
            
            # ATR trailing stop: exit if price rises 2.0*ATR above lowest low since entry
            trailing_stop = lowest_since_entry[i] + 2.0 * atr[i]
            
            # Exit conditions: trailing stop hit OR reversion to midpoint
            exit_condition = (close[i] > trailing_stop) or (close[i] > donchian_mid[i])
            
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