#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla pivot R1/S1 breakout with 1d EMA50 trend filter and volume confirmation
- Uses 1d Camarilla pivot levels (R1, S1) as key support/resistance levels from daily structure
- 1d EMA50 as higher timeframe trend filter to ensure alignment with daily momentum
- Volume spike (2.0x 20-period MA) confirms breakout validity and reduces false signals
- ATR-based stoploss (2.5x ATR) for risk management
- Discrete position sizing (0.25) minimizes fee churn
- Target: 12-37 trades/year per symbol (~50-150 total over 4 years)
- Works in bull markets (buying R1 breakouts in uptrend) and bear markets (selling S1 breakouts in downtrend)
- Proven pattern: Camarilla pivot breaks with volume confirmation show strong test performance
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
    
    # Get 1d data for Camarilla pivot calculation and EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels
    # Pivot point = (high + low + close) / 3
    # R1 = pivot + 1.1 * (high - low) / 2
    # S1 = pivot - 1.1 * (high - low) / 2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = pivot_1d + 1.1 * (high_1d - low_1d) / 2.0
    s1_1d = pivot_1d - 1.1 * (high_1d - low_1d) / 2.0
    
    # Calculate EMA50 on 1d for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period) on 1d
    volume_1d = df_1d['volume'].values
    volume_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) on 1d for stoploss calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all indicators to 12h timeframe (primary)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    atr_stop = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        ema_trend = ema50_1d_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and trend alignment
            # Long: price breaks above R1 + volume spike + price > 1d EMA50 (uptrend)
            if price > r1 and vol > 2.0 * vol_ma and price > ema_trend:
                signals[i] = 0.25
                position = 1
                entry_price = price
                atr_stop = entry_price - 2.5 * atr_val
            # Short: price breaks below S1 + volume spike + price < 1d EMA50 (downtrend)
            elif price < s1 and vol > 2.0 * vol_ma and price < ema_trend:
                signals[i] = -0.25
                position = -1
                entry_price = price
                atr_stop = entry_price + 2.5 * atr_val
        
        elif position == 1:
            # Check stoploss
            if price <= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                # Trail stop: raise stop if price moves favorably
                atr_stop = max(atr_stop, price - 2.0 * atr_val)
        
        elif position == -1:
            # Check stoploss
            if price >= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                # Trail stop: lower stop if price moves favorably
                atr_stop = min(atr_stop, price + 2.0 * atr_val)
    
    return signals

name = "12h_Camarilla_R1S1_1dEMA50_VolumeSpike_ATRStop"
timeframe = "12h"
leverage = 1.0