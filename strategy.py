# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Strategy: 6h_1w1d_AdaptiveChannel_VolumeRegime
Timeframe: 6h
Hypothesis: Combine adaptive price channel (based on ATR-scaled Donchian) with weekly bias and volume regime filter.
- Weekly trend filter (price above/below weekly VWAP) sets directional bias
- Daily volume regime (high/low volatility) adjusts channel sensitivity
- Adaptive channel: ATR-scaled Donchian breakouts with dynamic lookback
- Works in bull/bear by adapting volatility and using multi-timeframe alignment
Target: 20-50 trades/year, size 0.25
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for adaptive channel (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Adaptive Donchian channel: lookback adapts to volatility
    # Higher volatility = shorter lookback for faster reaction
    vol_ratio = atr / (pd.Series(atr).rolling(window=50, min_periods=50).mean().values + 1e-10)
    lookback = np.clip(20 * (2.0 - vol_ratio), 10, 30).astype(int)  # 10-30 periods
    
    # Calculate adaptive upper/lower bands using vectorized approach
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(n):
        lb = lookback[i]
        if i >= lb and not np.isnan(atr[i]):
            start_idx = i - lb + 1
            upper[i] = np.max(high[start_idx:i+1])
            lower[i] = np.min(low[start_idx:i+1])
    
    # Shift to use only completed bars
    upper = np.roll(upper, 1)
    lower = np.roll(lower, 1)
    upper[0] = np.nan
    lower[0] = np.nan
    
    # Weekly bias: price vs weekly VWAP
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly VWAP calculation
    typical_price_1w = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    vwap_1w = (typical_price_1w * df_1w['volume']).cumsum() / df_1w['volume'].cumsum()
    vwap_1w_values = vwap_1w.values
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w_values)
    
    # Daily volume regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily volume percentile (20-day lookback)
    vol_1d = df_1d['volume'].values
    vol_percentile = pd.Series(vol_1d).rolling(window=20, min_periods=20).rank(pct=True).values
    vol_percentile_aligned = align_htf_to_ltf(prices, df_1d, vol_percentile)
    
    # Volume regime: high vol (>70th percentile) = breakout mode, low vol (<30th) = mean reversion
    high_vol_regime = vol_percentile_aligned > 0.7
    low_vol_regime = vol_percentile_aligned < 0.3
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    start = max(30, 50)  # Ensure enough data for indicators
    for i in range(start, n):
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(vwap_1w_aligned[i]) or np.isnan(vol_percentile_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_regime_high = high_vol_regime[i]
        vol_regime_low = low_vol_regime[i]
        
        if position == 0:
            # Long conditions: bullish weekly bias + breakout above adaptive upper band
            # In high vol: breakout mode; in low vol: wait for stronger signal
            if price > vwap_1w_aligned[i]:  # Weekly bullish bias
                if vol_regime_high:
                    # High volatility: breakout entry
                    if price > upper[i]:
                        position = 1
                        signals[i] = position_size
                else:
                    # Low/medium volatility: require stronger breakout
                    if price > upper[i] * 1.002:  # 0.2% stronger breakout
                        position = 1
                        signals[i] = position_size
            
            # Short conditions: bearish weekly bias + breakdown below adaptive lower band
            elif price < vwap_1w_aligned[i]:  # Weekly bearish bias
                if vol_regime_high:
                    # High volatility: breakdown entry
                    if price < lower[i]:
                        position = -1
                        signals[i] = -position_size
                else:
                    # Low/medium volatility: require stronger breakdown
                    if price < lower[i] * 0.998:  # 0.2% stronger breakdown
                        position = -1
                        signals[i] = -position_size
        
        elif position == 1:
            # Exit long: weekly bias turns bearish OR price breaks below lower band
            if price < vwap_1w_aligned[i] or price < lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        
        elif position == -1:
            # Exit short: weekly bias turns bullish OR price breaks above upper band
            if price > vwap_1w_aligned[i] or price > upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w1d_AdaptiveChannel_VolumeRegime"
timeframe = "6h"
leverage = 1.0