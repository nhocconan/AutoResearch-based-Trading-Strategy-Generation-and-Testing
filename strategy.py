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
    
    # Get weekly HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(21) for trend filter
    ema_21_1w = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Get daily HTF data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily ATR(14) for volatility regime filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate daily Camarilla pivot levels (using prior day's OHLC)
    prior_high = df_1d['high'].shift(1).values
    prior_low = df_1d['low'].shift(1).values
    prior_close = df_1d['close'].shift(1).values
    
    camarilla_pivot = (prior_high + prior_low + prior_close) / 3.0
    camarilla_r3 = camarilla_pivot + 1.1 * (prior_high - prior_low)
    camarilla_s3 = camarilla_pivot - 1.1 * (prior_high - prior_low)
    camarilla_r4 = camarilla_pivot + 1.5 * (prior_high - prior_low)
    camarilla_s4 = camarilla_pivot - 1.5 * (prior_high - prior_low)
    
    # Align Camarilla levels to 6h
    camarilla_pivot_6h = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_r3_6h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_6h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_6h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_6h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate 6h ATR(14) for volatility entry filter
    tr1_6h = high - low
    tr2_6h = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3_6h = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    atr_14_6h = pd.Series(tr_6h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(camarilla_pivot_6h[i]) or np.isnan(camarilla_r3_6h[i]) or 
            np.isnan(camarilla_s3_6h[i]) or np.isnan(camarilla_r4_6h[i]) or 
            np.isnan(camarilla_s4_6h[i]) or np.isnan(atr_14_6h[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when daily ATR is elevated (> 0.8% of price)
        vol_regime = atr_14_1d_aligned[i] > 0.008 * close[i]
        
        # Trend filter: weekly EMA(21) direction
        weekly_uptrend = close[i] > ema_21_1w_aligned[i]
        weekly_downtrend = close[i] < ema_21_1w_aligned[i]
        
        # Long conditions:
        # 1. Weekly uptrend (price above weekly EMA21)
        # 2. Price breaks above Camarilla R3 with volume (bullish continuation)
        # 3. Volume confirmation: volume > 1.5x average (stricter)
        # 4. 6h ATR > 0.5% of price (ensure sufficient volatility)
        # 5. Daily volatility regime filter
        if (weekly_uptrend and
            close[i] > camarilla_r3_6h[i] and
            volume_ratio[i] > 1.5 and
            atr_14_6h[i] > 0.005 * close[i] and
            vol_regime):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Weekly downtrend (price below weekly EMA21)
        # 2. Price breaks below Camarilla S3 with volume (bearish continuation)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. 6h ATR > 0.5% of price
        # 5. Daily volatility regime filter
        elif (weekly_downtrend and
              close[i] < camarilla_s3_6h[i] and
              volume_ratio[i] > 1.5 and
              atr_14_6h[i] > 0.005 * close[i] and
              vol_regime):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WeeklyEMA_Camarilla_R3S3_Breakout_Volume_v1"
timeframe = "6h"
leverage = 1.0