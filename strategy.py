#!/usr/bin/env python3
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
    
    # Get daily HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA21 for trend filter
    ema21_1d = pd.Series(df_1d['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d)
    
    # Calculate daily ATR(14) for volatility regime filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate weekly HTF data for pivot levels (using prior week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly Camarilla pivot levels (R1/S1) from prior week
    prior_week_high = df_1w['high'].shift(1).values
    prior_week_low = df_1w['low'].shift(1).values
    prior_week_close = df_1w['close'].shift(1).values
    
    weekly_pivot = (prior_week_high + prior_week_low + prior_week_close) / 3.0
    weekly_r1 = weekly_pivot + 1.1 * (prior_week_high - prior_week_low)
    weekly_s1 = weekly_pivot - 1.1 * (prior_week_high - prior_week_low)
    
    # Align weekly Camarilla levels to 6h
    weekly_pivot_6h = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_6h = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_6h = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema21_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(weekly_pivot_6h[i]) or np.isnan(weekly_r1_6h[i]) or 
            np.isnan(weekly_s1_6h[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when daily ATR is elevated (> 0.8% of price)
        vol_regime = atr_14_1d_aligned[i] > 0.008 * close[i]
        
        # Trend filter: price above/below daily EMA21
        price_above_ema = close[i] > ema21_1d_aligned[i]
        price_below_ema = close[i] < ema21_1d_aligned[i]
        
        # Long conditions:
        # 1. Price above daily EMA21 (bullish bias)
        # 2. Price breaks above weekly R1 with volume (bullish continuation)
        # 3. Volume confirmation: volume > 1.8x average
        # 4. Daily volatility regime filter (avoid chop)
        if (price_above_ema and
            close[i] > weekly_r1_6h[i] and
            volume_ratio[i] > 1.8 and
            vol_regime):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below daily EMA21 (bearish bias)
        # 2. Price breaks below weekly S1 with volume (bearish continuation)
        # 3. Volume confirmation: volume > 1.8x average
        # 4. Daily volatility regime filter
        elif (price_below_ema and
              close[i] < weekly_s1_6h[i] and
              volume_ratio[i] > 1.8 and
              vol_regime):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Vol_Regime_EMA21_WeeklyCamarilla_R1S1_Breakout_v1"
timeframe = "6h"
leverage = 1.0