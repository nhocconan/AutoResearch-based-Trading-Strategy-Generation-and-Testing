#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla pivot (R1/S1) breakout + 1d EMA50 trend filter + volume confirmation + ATR trailing stop
- Camarilla pivot levels from 1d provide high-probability intraday support/resistance with proven edge on ETHUSDT
- 1d EMA50 ensures alignment with higher timeframe trend to avoid counter-trend trades
- Volume spike (1.8x 20-period MA) confirms institutional participation
- ATR(14) trailing stop (2.5x ATR) manages risk and reduces drawdown
- Discrete position sizing (0.25) minimizes fee churn
- Target: 15-25 trades/year per symbol (~60-100 total over 4 years)
- Works in bull markets (buying R1 breakouts in uptrend) and bear markets (selling S1 breakdowns in downtrend)
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
    
    # Get 1d data for Camarilla pivot calculation and EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 12h data for primary timeframe (volume, ATR)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate EMA50 on 1d for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla pivot levels (R1, S1) from 1d OHLC
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_range = high_1d - low_1d
    r1_1d = close_1d + (1.1 * camarilla_range) / 12
    s1_1d = close_1d - (1.1 * camarilla_range) / 12
    
    # Calculate ATR(14) on 12h for volatility and trailing stop
    def atr(high_arr, low_arr, close_arr, window=14):
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period TR is just high-low
        atr_vals = pd.Series(tr).rolling(window=window, min_periods=window).mean().values
        return atr_vals
    
    atr_14_12h = atr(high_12h, low_12h, close_12h, 14)
    
    # Volume average (20-period) on 12h
    volume_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    atr_14_aligned = align_htf_to_ltf(prices, df_12h, atr_14_12h)
    volume_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    highest_high_since_entry = 0.0  # For long trailing stop
    lowest_low_since_entry = 0.0    # For short trailing stop
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        ema_trend = ema50_1d_aligned[i]
        atr_val = atr_14_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and trend alignment
            # Long: price breaks above R1 + volume spike + price > 1d EMA50 (uptrend)
            if price > r1 and vol > 1.8 * vol_ma and price > ema_trend:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = price
            # Short: price breaks below S1 + volume spike + price < 1d EMA50 (downtrend)
            elif price < s1 and vol > 1.8 * vol_ma and price < ema_trend:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = price
        
        elif position == 1:
            # Update highest high for trailing stop
            if price > highest_high_since_entry:
                highest_high_since_entry = price
            
            # Exit long: price retracement to midpoint of R1-S1 OR ATR trailing stop
            mid_point = (r1 + s1) / 2.0
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
            
            # Exit short: price retracement to midpoint of R1-S1 OR ATR trailing stop
            mid_point = (r1 + s1) / 2.0
            trailing_stop = lowest_low_since_entry + 2.5 * atr_val
            
            if price > mid_point or price > trailing_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_1dEMA50_VolumeSpike_ATRStop"
timeframe = "12h"
leverage = 1.0