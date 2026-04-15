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
    
    # Get 12h HTF data once before loop (HTF trend and structure)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h Supertrend(ATR=10, mult=3.0) for trend filter
    # True Range
    tr1 = df_12h['high'] - df_12h['low']
    tr2 = np.abs(df_12h['high'] - np.concatenate([[df_12h['close'].iloc[0]], df_12h['close'].iloc[:-1]]))
    tr3 = np.abs(df_12h['low'] - np.concatenate([[df_12h['close'].iloc[0]], df_12h['close'].iloc[:-1]]))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_10_12h = pd.Series(tr_12h).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2_12h = (df_12h['high'] + df_12h['low']) / 2.0
    upper_band_12h = hl2_12h + (3.0 * atr_10_12h)
    lower_band_12h = hl2_12h - (3.0 * atr_10_12h)
    
    supertrend_12h = np.full_like(hl2_12h, np.nan, dtype=float)
    direction_12h = np.full_like(hl2_12h, 1, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(hl2_12h)):
        # Upper band
        if upper_band_12h[i] < upper_band_12h[i-1] or df_12h['close'].iloc[i-1] > upper_band_12h[i-1]:
            upper_band_12h[i] = upper_band_12h[i]
        else:
            upper_band_12h[i] = upper_band_12h[i-1]
            
        # Lower band
        if lower_band_12h[i] > lower_band_12h[i-1] or df_12h['close'].iloc[i-1] < lower_band_12h[i-1]:
            lower_band_12h[i] = lower_band_12h[i]
        else:
            lower_band_12h[i] = lower_band_12h[i-1]
            
        # Supertrend and direction
        if supertrend_12h[i-1] == upper_band_12h[i-1]:
            supertrend_12h[i] = lower_band_12h[i] if df_12h['close'].iloc[i] <= lower_band_12h[i] else upper_band_12h[i]
            direction_12h[i] = -1 if df_12h['close'].iloc[i] <= lower_band_12h[i] else 1
        else:
            supertrend_12h[i] = upper_band_12h[i] if df_12h['close'].iloc[i] >= upper_band_12h[i] else lower_band_12h[i]
            direction_12h[i] = 1 if df_12h['close'].iloc[i] >= upper_band_12h[i] else -1
    
    # Align Supertrend direction to 6h
    supertrend_dir_6h = align_htf_to_ltf(prices, df_12h, direction_12h.astype(float))
    
    # Get 1d HTF data for daily pivot points (support/resistance levels)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate daily pivot points (using prior day's OHLC)
    prior_high = df_1d['high'].shift(1).values
    prior_low = df_1d['low'].shift(1).values
    prior_close = df_1d['close'].shift(1).values
    
    # Standard pivot point
    pivot_point = (prior_high + prior_low + prior_close) / 3.0
    # Support and resistance levels
    r1 = 2 * pivot_point - prior_low
    s1 = 2 * pivot_point - prior_high
    r2 = pivot_point + (prior_high - prior_low)
    s2 = pivot_point - (prior_high - prior_low)
    r3 = prior_high + 2 * (pivot_point - prior_low)
    s3 = prior_low - 2 * (prior_high - pivot_point)
    
    # Align pivot levels to 6h
    pivot_point_6h = align_htf_to_ltf(prices, df_1d, pivot_point)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 6h ATR(14) for volatility filter and position sizing
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_dir_6h[i]) or np.isnan(pivot_point_6h[i]) or 
            np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or np.isnan(r2_6h[i]) or 
            np.isnan(s2_6h[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(atr_14[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 12h Supertrend uptrend (bullish HTF bias)
        # 2. Price above daily R1 with volume (break of minor resistance)
        # 3. Volume confirmation: volume > 1.3x average
        # 4. Volatility filter: ATR > 0.4% of price (avoid extremely low volatility)
        if (supertrend_dir_6h[i] == 1 and
            close[i] > r1_6h[i] and
            volume_ratio[i] > 1.3 and
            atr_14[i] > 0.004 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 12h Supertrend downtrend (bearish HTF bias)
        # 2. Price below daily S1 with volume (break of minor support)
        # 3. Volume confirmation: volume > 1.3x average
        # 4. Volatility filter: ATR > 0.4% of price
        elif (supertrend_dir_6h[i] == -1 and
              close[i] < s1_6h[i] and
              volume_ratio[i] > 1.3 and
              atr_14[i] > 0.004 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_12h_Supertrend_1d_Pivot_R1S1_Breakout_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0