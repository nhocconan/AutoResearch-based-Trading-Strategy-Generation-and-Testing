#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d EMA34 trend filter + volume confirmation + ATR trailing stop
- Uses Camarilla pivot levels (H3/L3) from 1d timeframe for high-probability breakout zones
- 1d EMA34 filter ensures trading only in direction of higher timeframe trend
- Volume confirmation (2.0x 20-period MA) reduces false breakouts
- ATR trailing stop (2.0x ATR) locks in profits and limits drawdown
- Fixed position size 0.25 to minimize fee churn
- Works in bull markets (buy H3 breakouts in uptrend) and bear markets (sell L3 breakdowns in downtrend)
- Proven pattern: Camarilla levels + volume + trend filter shows strong test performance on ETH/SOL
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
    
    # Get 1d data for Camarilla pivot levels and EMA34 (higher timeframe)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    hl_range = high_1d - low_1d
    H3 = close_1d + 1.1 * hl_range / 4
    L3 = close_1d - 1.1 * hl_range / 4
    # Shift by 1 to use only completed 1d bars (avoid look-ahead)
    H3_prev = np.roll(H3, 1)
    L3_prev = np.roll(L3, 1)
    H3_prev[0] = H3[0]
    L3_prev[0] = L3[0]
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume average (20-period) on 1d for confirmation
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) on 1d for stoploss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all 1d indicators to 4h timeframe (primary)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3_prev)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3_prev)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    atr_stop = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        H3_level = H3_aligned[i]
        L3_level = L3_aligned[i]
        ema34 = ema34_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and trend filter
            # Long: price breaks above H3 + volume spike + price > EMA34 (uptrend)
            if price > H3_level and vol > 2.0 * vol_ma and price > ema34:
                signals[i] = 0.25
                position = 1
                entry_price = price
                atr_stop = entry_price - 2.0 * atr_val
            # Short: price breaks below L3 + volume spike + price < EMA34 (downtrend)
            elif price < L3_level and vol > 2.0 * vol_ma and price < ema34:
                signals[i] = -0.25
                position = -1
                entry_price = price
                atr_stop = entry_price + 2.0 * atr_val
        
        elif position == 1:
            # Check stoploss
            if price <= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                # Trail stop: raise stop if price moves favorably
                atr_stop = max(atr_stop, price - 1.5 * atr_val)
        
        elif position == -1:
            # Check stoploss
            if price >= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                # Trail stop: lower stop if price moves favorably
                atr_stop = min(atr_stop, price + 1.5 * atr_val)
    
    return signals

name = "4h_Camarilla_H3L3_1dEMA34_VolumeSpike_ATRTrail"
timeframe = "4h"
leverage = 1.0