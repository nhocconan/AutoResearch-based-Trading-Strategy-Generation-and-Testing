#!/usr/bin/env python3
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
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) and EMA(200) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1w EMAs to 6h
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Get 1d HTF data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formula: Pivot = (H + L + C) / 3
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_hl = high_1d - low_1d
    
    # Resistance and Support levels
    r3 = pivot + (range_hl * 1.1 / 2)
    s3 = pivot - (range_hl * 1.1 / 2)
    r4 = pivot + (range_hl * 1.1)
    s4 = pivot - (range_hl * 1.1)
    
    # Align 1d pivot levels to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 6h ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter: EMA50 > EMA200 = bullish, EMA50 < EMA200 = bearish
        weekly_bullish = ema_50_1w_aligned[i] > ema_200_1w_aligned[i]
        weekly_bearish = ema_50_1w_aligned[i] < ema_200_1w_aligned[i]
        
        # Long conditions:
        # 1. Price breaks above Camarilla R4 with volume confirmation
        # 2. Weekly trend is bullish (EMA50 > EMA200)
        # 3. Volume > 2.0x average
        # 4. ATR > 0.5% of price (ensure sufficient volatility)
        if (close[i] > r4_aligned[i] and
            weekly_bullish and
            volume_ratio[i] > 2.0 and
            atr_14[i] > 0.005 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price breaks below Camarilla S4 with volume confirmation
        # 2. Weekly trend is bearish (EMA50 < EMA200)
        # 3. Volume > 2.0x average
        # 4. ATR > 0.5% of price
        elif (close[i] < s4_aligned[i] and
              weekly_bearish and
              volume_ratio[i] > 2.0 and
              atr_14[i] > 0.005 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_1w_EMA50_200_1d_Camarilla_R4S4_Breakout_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0