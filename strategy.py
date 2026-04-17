#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla H3/L3 breakout + 1w EMA50 trend filter + volume spike confirmation + ATR-based trailing stop
- Camarilla H3/L3 levels act as strong intraday support/resistance with proven edge on 1d timeframe
- 1w EMA50 ensures alignment with weekly trend to avoid counter-trend trades
- Volume spike (2.0x 20-period MA) confirms institutional participation
- ATR(14) trailing stop (3.0x ATR) manages risk and reduces drawdown
- Discrete position sizing (0.25) minimizes fee churn
- Target: 15-25 trades/year per symbol (~60-100 total over 4 years)
- Works in bull markets (buying H3 breakouts in uptrend) and bear markets (selling L3 breakdowns in downtrend)
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
    
    # Get 1d data for Camarilla, volume, ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Get 1w data for EMA50 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels on 1d
    def camarilla_levels(high_arr, low_arr, close_arr):
        # Classic Camarilla formula
        range_val = high_arr - low_arr
        close_prev = np.roll(close_arr, 1)
        close_prev[0] = close_arr[0]  # first bar uses current close
        
        H3 = close_prev + range_val * 1.1 / 4
        L3 = close_prev - range_val * 1.1 / 4
        H4 = close_prev + range_val * 1.1 / 2
        L4 = close_prev - range_val * 1.1 / 2
        
        return H3, L3, H4, L4
    
    H3_1d, L3_1d, H4_1d, L4_1d = camarilla_levels(high_1d, low_1d, close_1d)
    
    # Calculate ATR(14) on 1d for volatility and trailing stop
    def atr(high_arr, low_arr, close_arr, window=14):
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period TR is just high-low
        atr_vals = pd.Series(tr).rolling(window=window, min_periods=window).mean().values
        return atr_vals
    
    atr_14_1d = atr(high_1d, low_1d, close_1d, 14)
    
    # Volume average (20-period) on 1d
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 1d timeframe (prices is already 1d)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3_1d)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3_1d)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4_1d)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4_1d)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    highest_high_since_entry = 0.0  # For long trailing stop
    lowest_low_since_entry = 0.0    # For short trailing stop
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(H3_aligned[i]) or 
            np.isnan(L3_aligned[i]) or np.isnan(H4_aligned[i]) or 
            np.isnan(L4_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        H3 = H3_aligned[i]
        L3 = L3_aligned[i]
        H4 = H4_aligned[i]
        L4 = L4_aligned[i]
        ema_trend = ema50_1w_aligned[i]
        atr_val = atr_14_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and trend alignment
            # Long: price breaks above H3 + volume spike + price > 1w EMA50 (uptrend)
            if price > H3 and vol > 2.0 * vol_ma and price > ema_trend:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = price
            # Short: price breaks below L3 + volume spike + price < 1w EMA50 (downtrend)
            elif price < L3 and vol > 2.0 * vol_ma and price < ema_trend:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = price
        
        elif position == 1:
            # Update highest high for trailing stop
            if price > highest_high_since_entry:
                highest_high_since_entry = price
            
            # Exit long: price retracement to L3 level OR ATR trailing stop
            trailing_stop = highest_high_since_entry - 3.0 * atr_val
            
            if price < L3 or price < trailing_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest low for trailing stop
            if price < lowest_low_since_entry:
                lowest_low_since_entry = price
            
            # Exit short: price retracement to H3 level OR ATR trailing stop
            trailing_stop = lowest_low_since_entry + 3.0 * atr_val
            
            if price > H3 or price > trailing_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_H3L3_1wEMA50_VolumeSpike_ATRStop"
timeframe = "1d"
leverage = 1.0