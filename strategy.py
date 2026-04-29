#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above 20-day high, 1w EMA50 up-trend, volume > 2.0x average
# Short when price breaks below 20-day low, 1w EMA50 down-trend, volume > 2.0x average
# Exit when price reverts to 10-day midpoint (mean reversion)
# Uses discrete position sizing (0.25) and tight volume filter to limit trades to ~30-100 over 4 years.
# Uses 1w for signal direction/trend, 1d only for entry timing and breakout levels.
# This strategy targets 7-25 trades/year per symbol to avoid fee drag while capturing strong breakouts.
# Works in both bull (trend following) and bear (mean reversion exits) markets via volatility breakouts.

name = "1d_Donchian20_1wEMA50_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get 1d data for Donchian levels (based on previous 20 days)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-day Donchian levels using previous completed days only
    # Shift by 1 to use completed period only (no look-ahead)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # 20-day Donchian high and low
    donchian_high = pd.Series(prev_high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(prev_low).rolling(window=20, min_periods=20).min().values
    
    # 10-day midpoint for exit (mean reversion)
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align 1d indicators to 1d timeframe (prices index already matches)
    donchian_high_aligned = donchian_high  # Already aligned via get_htf_data
    donchian_low_aligned = donchian_low
    donchian_mid_aligned = donchian_mid
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Volume and 1w EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_dch_high = donchian_high_aligned[i]
        curr_dch_low = donchian_low_aligned[i]
        curr_dch_mid = donchian_mid_aligned[i]
        curr_ema50_1w = ema_50_1w_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price below 10-day midpoint (mean reversion)
            if curr_close < curr_dch_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above 10-day midpoint (mean reversion)
            if curr_close > curr_dch_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 20-period average (tight filter)
            vol_confirmed = curr_volume > 2.0 * curr_vol_ma
            
            # Long when price breaks above 20-day high, 1w EMA50 up-trend, volume confirmed
            if curr_high > curr_dch_high and curr_close > curr_ema50_1w and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below 20-day low, 1w EMA50 down-trend, volume confirmed
            elif curr_low < curr_dch_low and curr_close < curr_ema50_1w and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals