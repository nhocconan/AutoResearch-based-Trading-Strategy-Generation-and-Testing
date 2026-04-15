#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume confirmation and 4h trend filter
# Camarilla levels (R3/S3, R4/S4) act as intraday support/resistance.
# Breakout above R4 with volume confirmation and bullish 4h trend = long
# Breakdown below S4 with volume confirmation and bearish 4h trend = short
# Designed to work in both bull and bear markets via trend filter and volume confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data for Camarilla pivots (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate previous day's Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    # S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    range_1d = high_1d - low_1d
    r4 = close_1d + 1.5 * range_1d
    r3 = close_1d + 1.1 * range_1d
    s3 = close_1d - 1.1 * range_1d
    s4 = close_1d - 1.5 * range_1d
    
    # Align 1d Camarilla to 6h (wait for completed 1d bar)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Get 4h HTF data for trend filter (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 6h ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    # Pre-compute session filter (08-20 UTC) - avoid low-volume Asian session
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Start loop after warmup period
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 6h price breaks above Camarilla R4
        # 2. 4h EMA(50) trend filter: price above EMA50 (bullish bias)
        # 3. Volume confirmation: volume > 1.8x average
        # 4. Volatility filter: ATR > 0.4% of price (avoid low volatility chop)
        if (close[i] > r4_6h[i] and
            close[i] > ema_50_4h_aligned[i] and
            volume_ratio[i] > 1.8 and
            atr_14[i] > 0.004 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 6h price breaks below Camarilla S4
        # 2. 4h EMA(50) trend filter: price below EMA50 (bearish bias)
        # 3. Volume confirmation: volume > 1.8x average
        # 4. Volatility filter: ATR > 0.4% of price
        elif (close[i] < s4_6h[i] and
              close[i] < ema_50_4h_aligned[i] and
              volume_ratio[i] > 1.8 and
              atr_14[i] > 0.004 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Camarilla_R4S4_Breakout_Volume_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0