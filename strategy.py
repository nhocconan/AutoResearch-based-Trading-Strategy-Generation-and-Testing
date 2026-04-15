#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot reversal with 1d volume confirmation and 1w trend filter
# Designed for low trade frequency (target 12-37/year) with clear reversal logic
# Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets
# Uses Camarilla levels (H3/L3) from daily, volume spike for confirmation, and weekly EMA for trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data (primary timeframe) for price action
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Load 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Load 1d data for volume confirmation
    volume_1d = df_1d['volume'].values
    
    # Load 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels from previous 1d
    # H3 = close + 1.1 * (high - low) / 6
    # L3 = close - 1.1 * (high - low) / 6
    rango = high_1d - low_1d
    camarilla_h3 = close_1d + 1.1 * rango / 6
    camarilla_l3 = close_1d - 1.1 * rango / 6
    
    # Volume average (20-period on 1d)
    vol_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # EMA50 on 1w for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # ATR for volatility and stoploss (14-period on 12h)
    tr1 = np.maximum(high_12h[1:], low_12h[:-1]) - np.minimum(high_12h[1:], low_12h[:-1])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align all indicators to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    atr_aligned = align_htf_to_ltf(prices, df_12h, atr)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Base position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(vol_avg_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(atr_aligned[i])):
            continue
        
        # Volatility-adjusted position size (inverse vol)
        vol_factor = np.clip(0.5 * atr_aligned[i] / (close[i] + 1e-10), 0.5, 2.0)
        position_size = base_size / vol_factor
        position_size = np.clip(position_size, 0.15, 0.35)
        
        # Long entry: price touches or goes below L3 + uptrend + volume spike
        if (close[i] <= camarilla_l3_aligned[i] and 
            close[i] > ema50_1w_aligned[i] and 
            volume[i] > 1.8 * vol_avg_aligned[i] and 
            position <= 0):
            position = 1
            signals[i] = position_size
        
        # Short entry: price touches or goes above H3 + downtrend + volume spike
        elif (close[i] >= camarilla_h3_aligned[i] and 
              close[i] < ema50_1w_aligned[i] and 
              volume[i] > 1.8 * vol_avg_aligned[i] and 
              position >= 0):
            position = -1
            signals[i] = -position_size
        
        # Exit: reverse signal or price moves back to mean (pivot)
        elif position == 1 and (close[i] >= camarilla_h3_aligned[i] or 
                                close[i] < ema50_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] <= camarilla_l3_aligned[i] or 
                                 close[i] > ema50_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_1dVolume_1wEMA_Reversal"
timeframe = "12h"
leverage = 1.0