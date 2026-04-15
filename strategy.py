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
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d HTF data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (based on prior day)
    # Prior day's high, low, close
    prior_high = df_1d['high'].shift(1).values
    prior_low = df_1d['low'].shift(1).values
    prior_close = df_1d['close'].shift(1).values
    
    # Camarilla pivot: (H+L+C)/3
    camarilla_pivot = (prior_high + prior_low + prior_close) / 3.0
    # Camarilla R3: C + (H-L)*1.1/4
    camarilla_r3 = prior_close + (prior_high - prior_low) * 1.1 / 4
    # Camarilla S3: C - (H-L)*1.1/4
    camarilla_s3 = prior_close - (prior_high - prior_low) * 1.1 / 4
    
    # Align Camarilla levels to 6h
    camarilla_pivot_6h = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_r3_6h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_6h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
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
    
    # Session filter: avoid low-volume periods (22-02 UTC)
    hours = prices.index.hour
    in_session = (hours >= 2) & (hours <= 21)  # UTC 2-21, avoids Asian session lows
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_6h[i]) or np.isnan(camarilla_pivot_6h[i]) or 
            np.isnan(camarilla_r3_6h[i]) or np.isnan(camarilla_s3_6h[i]) or 
            np.isnan(atr_14[i]) or np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 6h price above 12h EMA50 (bullish trend)
        # 2. Price touches or breaks above Camarilla R3 (strong bullish breakout)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        if (close[i] > ema_50_6h[i] and
            close[i] >= camarilla_r3_6h[i] and
            volume_ratio[i] > 1.5 and
            atr_14[i] > 0.005 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 6h price below 12h EMA50 (bearish trend)
        # 2. Price touches or breaks below Camarilla S3 (strong bearish breakdown)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility filter: ATR > 0.5% of price
        elif (close[i] < ema_50_6h[i] and
              close[i] <= camarilla_s3_6h[i] and
              volume_ratio[i] > 1.5 and
              atr_14[i] > 0.005 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_12h_EMA50_1d_CamarillaR3S3_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0