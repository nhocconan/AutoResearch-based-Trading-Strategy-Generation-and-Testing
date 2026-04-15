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
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (based on prior day)
    # Camarilla uses previous day's OHLC
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3.0
    camarilla_r3 = camarilla_pivot + 1.1 * (prev_high - prev_low) / 2.0
    camarilla_s3 = camarilla_pivot - 1.1 * (prev_high - prev_low) / 2.0
    camarilla_r4 = camarilla_pivot + 1.1 * (prev_high - prev_low)
    camarilla_s4 = camarilla_pivot - 1.1 * (prev_high - prev_low)
    
    # Align 1d Camarilla to 6h
    camarilla_pivot_6h = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_r3_6h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_6h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_6h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_6h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Get 1w HTF data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend direction
    weekly_ema20 = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_ema20_6h = align_htf_to_ltf(prices, df_1w, weekly_ema20)
    
    # Calculate 6h ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    # Precompute session filter (00-24 UTC for 6h - less restrictive)
    hours = prices.index.hour
    in_session = (hours >= 0) & (hours <= 23)  # Always true for 6h, kept for structure
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_pivot_6h[i]) or np.isnan(camarilla_r3_6h[i]) or 
            np.isnan(camarilla_s3_6h[i]) or np.isnan(camarilla_r4_6h[i]) or 
            np.isnan(camarilla_s4_6h[i]) or np.isnan(weekly_ema20_6h[i]) or 
            np.isnan(atr_14[i]) or np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 6h price breaks above Camarilla R3 (strong intraday resistance)
        # 2. Weekly trend filter: price above weekly EMA20 (bullish weekly bias)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        if (close[i] > camarilla_r3_6h[i] and
            close[i] > weekly_ema20_6h[i] and
            volume_ratio[i] > 1.5 and
            atr_14[i] > 0.005 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 6h price breaks below Camarilla S3 (strong intraday support)
        # 2. Weekly trend filter: price below weekly EMA20 (bearish weekly bias)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility filter: ATR > 0.5% of price
        elif (close[i] < camarilla_s3_6h[i] and
              close[i] < weekly_ema20_6h[i] and
              volume_ratio[i] > 1.5 and
              atr_14[i] > 0.005 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_1d_CamarillaR3S3_1w_EMA20_Volume_ATR_Filter_v1"
timeframe = "6h"
leverage = 1.0