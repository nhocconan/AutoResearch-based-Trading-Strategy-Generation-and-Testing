#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume spike confirmation
- Long when price breaks above 1d Donchian upper (20-period high) AND price > 1w EMA34 AND volume > 2.0x 20-period average
- Short when price breaks below 1d Donchian lower (20-period low) AND price < 1w EMA34 AND volume > 2.0x 20-period average
- Exit when price crosses the 1d Donchian midpoint (mean reversion to median)
- Uses 1w EMA34 for HTF trend alignment to avoid counter-trend entries
- Volume spike ensures institutional participation and reduces false breakouts
- Designed for both bull and bear markets: trend filter prevents counter-trend entries
- Target: 7-25 trades/year (30-100 total over 4 years) to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1w EMA34
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Get 1d data for Donchian levels (primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (based on previous bar's OHLC to avoid look-ahead)
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    prev_close = df_1d['close'].shift(1)
    
    donchian_upper = prev_high.rolling(window=20, min_periods=20).max()
    donchian_lower = prev_low.rolling(window=20, min_periods=20).min()
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Convert to numpy arrays and align to 1d timeframe
    donchian_upper_vals = donchian_upper.values
    donchian_lower_vals = donchian_lower.values
    donchian_mid_vals = donchian_mid.values
    
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_vals)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_vals)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid_vals)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 35, 21)  # Need 20 for Donchian, 35 for EMA34 (34+1), 1 for shifted OHLC
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions (using 1d Donchian levels)
        breakout_up = close[i] > donchian_upper_aligned[i]  # Break above Donchian upper
        breakout_down = close[i] < donchian_lower_aligned[i]  # Break below Donchian lower
        
        # Trend filter (using 1w EMA34)
        uptrend = close[i] > ema34_1w_aligned[i]
        downtrend = close[i] < ema34_1w_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: bullish breakout + uptrend + volume confirmation
            if breakout_up and uptrend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout + downtrend + volume confirmation
            elif breakout_down and downtrend and volume_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price crosses 1d Donchian midpoint (mean reversion)
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below midpoint
                if close[i] < donchian_mid_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: price crosses above midpoint
                if close[i] > donchian_mid_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0