#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h price with 1-day volume profile (VWAP and volume clusters).
# In ranging markets, price reverts to value area (VWAP ± 1σ).
# In trending markets, breaks through value area with volume expansion.
# Uses 1-day volume profile calculated from actual volume at price levels.
# Adaptive to bull/bear via 1-week trend filter (price vs 20-period EMA).
# Target: 20-30 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for volume profile
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily VWAP and standard deviation using actual volume at price
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Typical price
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    
    # VWAP = sum(price * volume) / sum(volume)
    vwap_num = np.cumsum(typical_price * volume_1d)
    vwap_den = np.cumsum(volume_1d)
    vwap = np.where(vwap_den != 0, vwap_num / vwap_den, typical_price)
    
    # Variance for standard deviation
    var_num = np.cumsum((typical_price - vwap) ** 2 * volume_1d)
    variance = np.where(vwap_den != 0, var_num / vwap_den, 0)
    std_dev = np.sqrt(np.maximum(variance, 0))
    
    # Value area: VWAP ± 1 standard deviation
    va_high = vwap + std_dev
    va_low = vwap - std_dev
    
    # Align VWAP and value area to 6h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap)
    va_high_aligned = align_htf_to_ltf(prices, df_1d, va_high)
    va_low_aligned = align_htf_to_ltf(prices, df_1d, va_low)
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA(20) for trend
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # 6h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 6h volume ratio (current / 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma_20 == 0, 1, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(vwap_aligned[i]) or np.isnan(va_high_aligned[i]) or np.isnan(va_low_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vwap_val = vwap_aligned[i]
        va_high_val = va_high_aligned[i]
        va_low_val = va_low_aligned[i]
        ema_trend = ema_20_1w_aligned[i]
        atr = atr_14[i]
        vol_ratio_6h = vol_ratio[i]
        
        # Determine market regime from weekly trend
        uptrend = price > ema_trend
        downtrend = price < ema_trend
        
        # Volatility filter: avoid extreme volatility
        atr_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values[i]
        vol_filter = (atr < 3.0 * atr_ma_20)
        
        # Volume filter: require above-average volume
        vol_filter = vol_filter and (vol_ratio_6h > 1.3)
        
        if position == 0:
            # In uptrend: look for long near VA low (value area support)
            if uptrend and vol_filter:
                if price <= va_low_val * 1.005:  # Near VA low with small buffer
                    signals[i] = 0.25
                    position = 1
            # In downtrend: look for short near VA high (value area resistance)
            elif downtrend and vol_filter:
                if price >= va_high_val * 0.995:  # Near VA high with small buffer
                    signals[i] = -0.25
                    position = -1
            # In ranging (no clear trend): mean reversion at value area edges
            else:
                if price <= va_low_val * 1.005 and vol_filter:  # At VA low
                    signals[i] = 0.25
                    position = 1
                elif price >= va_high_val * 0.995 and vol_filter:  # At VA high
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price reaches VWAP or value area high, or filters fail
            if price >= vwap_val or price >= va_high_val or (not vol_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches VWAP or value area low, or filters fail
            if price <= vwap_val or price <= va_low_val or (not vol_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_VolumeProfile_VWAP_ValueArea_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0