#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d chart with weekly pivot points (R1/S1) for mean reversion, confirmed by weekly trend filter.
# In ranging markets, price reverts to pivot; in trending markets, breaks through R1/S1 with volume.
# Works in bull/bear by adapting to regime via weekly trend filter.
# Target: 15-25 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points (standard formula)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot = (H + L + C)/3
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # R1 = 2*P - L, S1 = 2*P - H
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # Align pivot levels to 1d timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA(20) for trend
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # 1d data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d ATR(14) for volatility and stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d volume ratio (current / 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma_20 == 0, 1, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_trend = ema_20_1w_aligned[i]
        atr = atr_14[i]
        vol_ratio_1d = vol_ratio[i]
        
        # Determine market regime from weekly trend
        uptrend = price > ema_trend
        downtrend = price < ema_trend
        
        # Volatility filter: avoid extreme volatility
        atr_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values[i]
        vol_filter = (atr < 3.0 * atr_ma_20)
        
        # Volume filter: require above-average volume
        vol_filter = vol_filter and (vol_ratio_1d > 1.5)
        
        if position == 0:
            # In uptrend: look for long near S1 (support)
            if uptrend and vol_filter:
                if price <= s1_val * 1.005:  # Near S1 with small buffer
                    signals[i] = 0.25
                    position = 1
            # In downtrend: look for short near R1 (resistance)
            elif downtrend and vol_filter:
                if price >= r1_val * 0.995:  # Near R1 with small buffer
                    signals[i] = -0.25
                    position = -1
            # In ranging (no clear trend): fade extremes
            else:
                if price <= s1_val * 1.005 and vol_filter:  # Near S1
                    signals[i] = 0.25
                    position = 1
                elif price >= r1_val * 0.995 and vol_filter:  # Near R1
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price reaches pivot or stops reversed
            if price >= pivot_val or (not vol_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches pivot or stops reversed
            if price <= pivot_val or (not vol_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Pivot_R1S1_MeanReversion_TrendFilter_v1"
timeframe = "1d"
leverage = 1.0