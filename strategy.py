#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter (EMA34) and volume confirmation
# - Long when price breaks above Donchian upper band (20-period high) AND 1d EMA34 > prior EMA34 (uptrend) AND volume > 2.0x 20-period volume SMA
# - Short when price breaks below Donchian lower band (20-period low) AND 1d EMA34 < prior EMA34 (downtrend) AND volume > 2.0x 20-period volume SMA
# - Exit: ATR trailing stop (3.0x ATR) or reversion to Donchian midpoint
# - Uses 1d for trend filter (avoid counter-trend trades) and 4h for precise entry/exit timing
# - Position sizing: 0.25 discrete level to control drawdown and minimize fee churn
# - Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag while maintaining statistical significance
# - Donchian breakouts capture strong momentum moves; volume confirmation filters false breakouts; 1d EMA trend filter ensures alignment with higher timeframe momentum

name = "4h_1d_donchian_volume_trend_v1"
timeframe = "4h"
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
    open_time = prices['open_time'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:  # Need sufficient data for EMA34
        return signals
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Trend: rising EMA (bullish) or falling EMA (bearish)
    ema_34_rising = ema_34_1d > np.roll(ema_34_1d, 1)
    ema_34_falling = ema_34_1d < np.roll(ema_34_1d, 1)
    # Handle first value (no previous EMA)
    ema_34_rising[0] = False
    ema_34_falling[0] = False
    # Align to 4h timeframe with proper delay (completed 1d bar only)
    ema_34_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_34_rising)
    ema_34_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_34_falling)
    
    # Calculate ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period) for breakout signals
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high_20 + lowest_low_20) / 2.0
    
    # Calculate 20-period volume SMA for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Track highest high since entry for trailing stop (long)
    # Track lowest low since entry for trailing stop (short)
    highest_since_entry = np.full(n, np.nan)
    lowest_since_entry = np.full(n, np.nan)
    
    for i in range(20, n):  # Start from 20 to have sufficient lookback
        # Skip if any required data is invalid
        if (np.isnan(atr[i]) or np.isnan(donchian_mid[i]) or np.isnan(volume_sma_20[i]) or
            np.isnan(ema_34_rising_aligned[i]) or np.isnan(ema_34_falling_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 4h volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > 2.0 * volume_sma_20[i]
        
        # Donchian breakout signals
        breakout_up = close[i] > highest_high_20[i-1]  # Break above upper band
        breakout_down = close[i] < lowest_low_20[i-1]  # Break below lower band
        
        if position == 0:  # Flat - look for entry
            # Long: price breaks above upper band AND 1d EMA rising AND volume confirmation
            if breakout_up and ema_34_rising_aligned[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
                highest_since_entry[i] = high[i]  # Initialize trailing stop
            # Short: price breaks below lower band AND 1d EMA falling AND volume confirmation
            elif breakout_down and ema_34_falling_aligned[i] and vol_confirm:
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
            
            # ATR trailing stop: exit if price drops 3.0*ATR below highest high since entry
            trailing_stop = highest_since_entry[i] - 3.0 * atr[i]
            
            # Exit conditions: trailing stop hit OR reversion to Donchian midpoint
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
            
            # ATR trailing stop: exit if price rises 3.0*ATR above lowest low since entry
            trailing_stop = lowest_since_entry[i] + 3.0 * atr[i]
            
            # Exit conditions: trailing stop hit OR reversion to Donchian midpoint
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