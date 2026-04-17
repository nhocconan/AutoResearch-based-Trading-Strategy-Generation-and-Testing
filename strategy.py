#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot R3/S3 breakout + 1w EMA34 trend filter + volume confirmation + ATR trailing stop
- Uses 1w EMA34 as HTF trend filter for stronger regime identification (more stable than daily)
- Camarilla R3/S3 breakout captures momentum with proven edge on ETHUSDT (top performer)
- Volume spike (2.0x 20-period MA) confirms institutional participation
- ATR(14) trailing stop (2.5x ATR) manages risk with tighter stop for better win rate
- Discrete position sizing (0.25) minimizes fee churn
- Target: 20-30 trades/year per symbol (~80-120 total over 4 years)
- Works in bull markets (buying R3 breakouts in uptrend) and bear markets (selling S3 breakdowns in downtrend)
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
    close_1w = df_1w['close'].values
    
    # Get 4h data for primary timeframe (Camarilla, volume, ATR)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    open_4h = df_4h['open'].values
    
    # Calculate EMA34 on 1w for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla levels (based on previous 4h bar) on 4h
    def camarilla_levels(high_arr, low_arr, close_arr):
        # Typical price from previous bar
        typical_price = (high_arr[:-1] + low_arr[:-1] + close_arr[:-1]) / 3.0
        range_val = high_arr[:-1] - low_arr[:-1]
        
        # Shift to align with current bar (previous bar's levels)
        typical_price = np.concatenate([[np.nan], typical_price])
        range_val = np.concatenate([[np.nan], range_val])
        
        # Camarilla levels
        R3 = typical_price + range_val * 1.1/2
        R2 = typical_price + range_val * 1.1/4
        R1 = typical_price + range_val * 1.1/6
        S1 = typical_price - range_val * 1.1/6
        S2 = typical_price - range_val * 1.1/4
        S3 = typical_price - range_val * 1.1/2
        
        return R3, R2, R1, S1, S2, S3
    
    R3_4h, R2_4h, R1_4h, S1_4h, S2_4h, S3_4h = camarilla_levels(high_4h, low_4h, close_4h)
    
    # Calculate ATR(14) on 4h for volatility and trailing stop
    def atr(high_arr, low_arr, close_arr, window=14):
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period TR is just high-low
        atr_vals = pd.Series(tr).rolling(window=window, min_periods=window).mean().values
        return atr_vals
    
    atr_14_4h = atr(high_4h, low_4h, close_4h, 14)
    
    # Volume average (20-period) on 4h
    volume_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    R3_aligned = align_htf_to_ltf(prices, df_4h, R3_4h)
    R2_aligned = align_htf_to_ltf(prices, df_4h, R2_4h)
    R1_aligned = align_htf_to_ltf(prices, df_4h, R1_4h)
    S1_aligned = align_htf_to_ltf(prices, df_4h, S1_4h)
    S2_aligned = align_htf_to_ltf(prices, df_4h, S2_4h)
    S3_aligned = align_htf_to_ltf(prices, df_4h, S3_4h)
    atr_14_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    highest_high_since_entry = 0.0  # For long trailing stop
    lowest_low_since_entry = 0.0    # For short trailing stop
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        R3 = R3_aligned[i]
        S3 = S3_aligned[i]
        ema_trend = ema34_1w_aligned[i]
        atr_val = atr_14_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and trend alignment
            # Long: price breaks above R3 + volume spike + price > 1w EMA34 (uptrend)
            if price > R3 and vol > 2.0 * vol_ma and price > ema_trend:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = price
            # Short: price breaks below S3 + volume spike + price < 1w EMA34 (downtrend)
            elif price < S3 and vol > 2.0 * vol_ma and price < ema_trend:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = price
        
        elif position == 1:
            # Update highest high for trailing stop
            if price > highest_high_since_entry:
                highest_high_since_entry = price
            
            # Exit long: price retracement to R2 level OR ATR trailing stop
            trailing_stop = highest_high_since_entry - 2.5 * atr_val
            
            if price < R2 or price < trailing_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest low for trailing stop
            if price < lowest_low_since_entry:
                lowest_low_since_entry = price
            
            # Exit short: price retracement to S2 level OR ATR trailing stop
            trailing_stop = lowest_low_since_entry + 2.5 * atr_val
            
            if price > S2 or price > trailing_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_1wEMA34_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0