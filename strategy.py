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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Elder Ray indicators
    # Bull Power = High - EMA13(Close)
    # Bear Power = Low - EMA13(Close)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # EMA13 on 1d close
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power and Bear Power
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align Elder Ray to 6h
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Get 1w HTF data for regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w ADX(14) for trend strength
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr3 = np.abs(low_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.concatenate([[high_1w[0]], high_1w[:-1]])) > 
                       (np.concatenate([[low_1w[0]], low_1w[:-1]]) - low_1w), 
                       np.maximum(high_1w - np.concatenate([[high_1w[0]], high_1w[:-1]]), 0), 0)
    dm_minus = np.where((np.concatenate([[low_1w[0]], low_1w[:-1]]) - low_1w) > 
                        (high_1w - np.concatenate([[high_1w[0]], high_1w[:-1]])), 
                        np.maximum(np.concatenate([[low_1w[0]], low_1w[:-1]]) - low_1w, 0), 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus_14 = 100 * dm_plus_14 / (tr_14 + 1e-10)
    di_minus_14 = 100 * dm_minus_14 / (tr_14 + 1e-10)
    
    # DX and ADX
    dx_14 = 100 * np.abs(di_plus_14 - di_minus_14) / (di_plus_14 + di_minus_14 + 1e-10)
    adx_14 = pd.Series(dx_14).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1w ADX to 6h
    adx_6h = align_htf_to_ltf(prices, df_1w, adx_14)
    
    # Calculate 6h ATR(14) for volatility filter
    tr1_6h = high - low
    tr2_6h = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3_6h = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    atr_14_6h = pd.Series(tr_6h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    # Session filter: UTC 0-23 (all 6h bars)
    hours = prices.index.hour
    in_session = (hours >= 0) & (hours <= 23)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or 
            np.isnan(adx_6h[i]) or np.isnan(atr_14_6h[i]) or 
            np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Bull Power > 0 (buying pressure)
        # 2. Bear Power < 0 (weak selling pressure)
        # 3. ADX > 25 (trending market)
        # 4. Volume confirmation: volume > 1.5x average
        # 5. Volatility filter: ATR > 0.3% of price
        if (bull_power_6h[i] > 0 and
            bear_power_6h[i] < 0 and
            adx_6h[i] > 25 and
            volume_ratio[i] > 1.5 and
            atr_14_6h[i] > 0.003 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Bear Power < 0 (selling pressure)
        # 2. Bull Power < 0 (weak buying pressure)
        # 3. ADX > 25 (trending market)
        # 4. Volume confirmation: volume > 1.5x average
        # 5. Volatility filter: ATR > 0.3% of price
        elif (bear_power_6h[i] < 0 and
              bull_power_6h[i] < 0 and
              adx_6h[i] > 25 and
              volume_ratio[i] > 1.5 and
              atr_14_6h[i] > 0.003 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_1d_ElderRay_1w_ADX_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0