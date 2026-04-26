#!/usr/bin/env python3
"""
1h_VolumeSpike_MeanReversion_4hTrend_v1
Hypothesis: On 1h timeframe, fade extreme volume spikes when price deviates from 4h VWAP, with 4h trend filter to avoid counter-trend trades in strong markets. Volume spike (>2.5x median) indicates exhaustion; mean reversion to 4h VWAP provides edge. 4h EMA50 filter ensures we only trade with higher timeframe momentum. Designed for low trade frequency (15-30/year) to minimize fee drag in difficult 1h timeframe. Works in bull/bear by aligning with 4h trend direction.
"""

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
    
    # Get 4h data for HTF trend and VWAP
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 4h VWAP (volume-weighted average price)
    typical_price_4h = (df_4h['high'].values + df_4h['low'].values + df_4h['close'].values) / 3.0
    vwap_4h = (typical_price_4h * df_4h['volume'].values).cumsum() / df_4h['volume'].values.cumsum()
    vwap_4h = vwap_4h.values
    
    # 1h volume median for spike detection (30-period)
    vol_median = pd.Series(volume).rolling(window=30, min_periods=30).median().values
    
    # ATR(14) for volatility stops
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    vwap_4h_aligned = align_htf_to_ltf(prices, df_4h, vwap_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of EMA(50) 4h (50), VWAP (need cumulative data), volume median (30), ATR (14)
    start_idx = max(50, 1, 30, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vwap_4h_aligned[i]) or
            np.isnan(vol_median[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        ema_50_4h_val = ema_50_4h_aligned[i]
        vwap_4h_val = vwap_4h_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        atr_val = atr[i]
        
        # Trend filter: uptrend if price > EMA50, downtrend if price < EMA50
        uptrend = close_val > ema_50_4h_val
        downtrend = close_val < ema_50_4h_val
        
        # Volume spike filter: extreme volume indicates exhaustion
        volume_spike = volume_val > 2.5 * vol_median_val
        
        # Deviation from 4h VWAP: positive = price above VWAP, negative = below
        vwap_deviation = (close_val - vwap_4h_val) / vwap_4h_val
        
        if position == 0:
            # Long: price below VWAP (oversold) + volume spike + uptrend (fade with trend)
            long_signal = (vwap_deviation < -0.015) and \
                          volume_spike and \
                          uptrend
            
            # Short: price above VWAP (overbought) + volume spike + downtrend (fade with trend)
            short_signal = (vwap_deviation > 0.015) and \
                           volume_spike and \
                           downtrend
            
            if long_signal:
                signals[i] = 0.20
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.20
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit conditions: price reverts to VWAP or stop loss
            if close_val >= vwap_4h_val or close_val < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit conditions: price reverts to VWAP or stop loss
            if close_val <= vwap_4h_val or close_val > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_VolumeSpike_MeanReversion_4hTrend_v1"
timeframe = "1h"
leverage = 1.0