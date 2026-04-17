#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout + 1w EMA34 trend filter + volume confirmation + ATR trailing stop
- Uses 1w EMA34 as HTF trend filter to capture weekly momentum
- Donchian(20) breakout on 1d timeframe for precise entry timing
- Volume spike (2.0x 20-period MA) confirms institutional participation
- ATR(14) trailing stop (3.0x ATR) manages risk
- Discrete position sizing (0.25) minimizes fee churn
- Target: 10-25 trades/year per symbol (~40-100 total over 4 years)
- Works in bull markets (buying upper band breakouts in uptrend) and bear markets (selling lower band breakdowns in downtrend)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA34 on 1w for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Donchian channels (20-period) on 1d
    def donchian_channels(high_arr, low_arr, window=20):
        upper = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    donchian_upper_1d, donchian_lower_1d = donchian_channels(high, low, 20)
    
    # Calculate ATR(14) on 1d for volatility and trailing stop
    def atr(high_arr, low_arr, close_arr, window=14):
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period TR is just high-low
        atr_vals = pd.Series(tr).rolling(window=window, min_periods=window).mean().values
        return atr_vals
    
    atr_14_1d = atr(high, low, close, 14)
    
    # Volume average (20-period) on 1d
    volume_ma_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 1d timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper_1d)  # Using 1w df for alignment but 1d values
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower_1d)
    atr_14_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1d)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    highest_high_since_entry = 0.0  # For long trailing stop
    lowest_low_since_entry = 0.0    # For short trailing stop
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        ema_trend = ema34_1w_aligned[i]
        atr_val = atr_14_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and trend alignment
            # Long: price breaks above upper Donchian + volume spike + price > 1w EMA34 (uptrend)
            if price > upper and vol > 2.0 * vol_ma and price > ema_trend:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = price
            # Short: price breaks below lower Donchian + volume spike + price < 1w EMA34 (downtrend)
            elif price < lower and vol > 2.0 * vol_ma and price < ema_trend:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = price
        
        elif position == 1:
            # Update highest high for trailing stop
            if price > highest_high_since_entry:
                highest_high_since_entry = price
            
            # Exit long: price retracement to midpoint of Donchian channel OR ATR trailing stop
            mid_point = (upper + lower) / 2.0
            trailing_stop = highest_high_since_entry - 3.0 * atr_val
            
            if price < mid_point or price < trailing_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest low for trailing stop
            if price < lowest_low_since_entry:
                lowest_low_since_entry = price
            
            # Exit short: price retracement to midpoint of Donchian channel OR ATR trailing stop
            mid_point = (upper + lower) / 2.0
            trailing_stop = lowest_low_since_entry + 3.0 * atr_val
            
            if price > mid_point or price > trailing_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA34_VolumeSpike_ATRStop"
timeframe = "1d"
leverage = 1.0