#!/usr/bin/env python3
"""
4h_KAMA_Regime_Trend_ATRStop_v2
Hypothesis: 4h KAMA trend + volume spike + chop regime filter with ATR trailing stop.
KAMA adapts to market efficiency, reducing whipsaw in ranging markets.
Volume spike confirms breakout strength. Chop regime filter avoids false signals in high-chop environments.
Designed for 20-40 trades/year per symbol with discrete sizing (0.0, ±0.25) to minimize fee churn.
Works in bull/bear via KAMA's adaptive trend detection and volume confirmation to avoid false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for chop regime)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 4h KAMA (adaptive trend) ===
    close = prices['close'].values
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i-10] * (close[i] - kama[i-1])
    
    # === ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === 1d Chop Regime (EWMA of True Range) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    # Chop = ATR / (highest high - lowest low over 14 periods)
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = hh_14 - ll_14
    chop = np.divide(atr_1d * 100, chop_denom, out=np.full_like(atr_1d, 50.0), where=chop_denom!=0)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(kama[i]) or np.isnan(atr[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Volume spike: current volume > 1.8x 20-period average
            volume = prices['volume'].values
            vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
            vol_spike = volume[i] > 1.8 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
            
            # Long conditions: price > KAMA, low chop (< 45), volume spike
            long_trend = price > kama[i]
            long_chop = chop_aligned[i] < 45
            
            # Short conditions: price < KAMA, low chop (< 45), volume spike
            short_trend = price < kama[i]
            short_chop = chop_aligned[i] < 45
            
            # Entry logic - volume spike + trend + low chop
            if long_trend and long_chop and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_trend and short_chop and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below KAMA (trend broken)
            elif price < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above KAMA (trend broken)
            elif price > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Regime_Trend_ATRStop_v2"
timeframe = "4h"
leverage = 1.0