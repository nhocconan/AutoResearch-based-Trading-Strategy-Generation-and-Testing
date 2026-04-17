#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout + 1d EMA50 trend filter + volume spike confirmation + ATR trailing stop
- Uses 1d EMA50 as HTF trend filter for robust trend identification
- Camarilla R1/S1 levels from 1d provide precise intraday support/resistance
- Volume spike (2.0x 20-period MA) confirms institutional participation
- ATR(14) trailing stop (2.5x ATR) manages risk with discretionary exits
- Discrete position sizing (0.25) minimizes fee churn
- Target: 15-25 trades/year per symbol (~60-100 total over 4 years)
- Works in bull markets (buying R1 breakouts in uptrend) and bear markets (selling S1 breakdowns in downtrend)
- Aligns with proven patterns: Camarilla pivot strategies show strong test performance when properly filtered
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
    
    # Get 1d data for HTF indicators (EMA50, Camarilla pivots)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Get 4h data for primary timeframe (volume, ATR, price)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate EMA50 on 1d for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla pivot levels (R1, S1) from 1d OHLC
    def camarilla_levels(high_arr, low_arr, close_arr):
        typical = (high_arr + low_arr + close_arr) / 3.0
        range_val = high_arr - low_arr
        R1 = close_arr + range_val * 1.1 / 12.0
        S1 = close_arr - range_val * 1.1 / 12.0
        return R1, S1
    
    camarilla_R1_1d, camarilla_S1_1d = camarilla_levels(high_1d, low_1d, close_1d)
    
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
    
    # Align all HTF indicators to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1_1d)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1_1d)
    atr_14_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    highest_high_since_entry = 0.0  # For long trailing stop
    lowest_low_since_entry = 0.0    # For short trailing stop
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(camarilla_R1_aligned[i]) or 
            np.isnan(camarilla_S1_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        R1 = camarilla_R1_aligned[i]
        S1 = camarilla_S1_aligned[i]
        ema_trend = ema50_1d_aligned[i]
        atr_val = atr_14_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and trend alignment
            # Long: price breaks above R1 + volume spike + price > 1d EMA50 (uptrend)
            if price > R1 and vol > 2.0 * vol_ma and price > ema_trend:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = price
            # Short: price breaks below S1 + volume spike + price < 1d EMA50 (downtrend)
            elif price < S1 and vol > 2.0 * vol_ma and price < ema_trend:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = price
        
        elif position == 1:
            # Update highest high for trailing stop
            if price > highest_high_since_entry:
                highest_high_since_entry = price
            
            # Exit long: price retracement to midpoint between R1 and S1 OR ATR trailing stop
            mid_point = (R1 + S1) / 2.0
            trailing_stop = highest_high_since_entry - 2.5 * atr_val
            
            if price < mid_point or price < trailing_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest low for trailing stop
            if price < lowest_low_since_entry:
                lowest_low_since_entry = price
            
            # Exit short: price retracement to midpoint between R1 and S1 OR ATR trailing stop
            mid_point = (R1 + S1) / 2.0
            trailing_stop = lowest_low_since_entry + 2.5 * atr_val
            
            if price > mid_point or price > trailing_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_1dEMA50_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0