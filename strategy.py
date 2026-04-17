#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using weekly Camarilla pivot levels (H3/L3) with 1d EMA34 trend filter and volume confirmation.
- Long when price closes above weekly H3 + volume > 1.5x 20-period 6h volume MA + price above 1d EMA34
- Short when price closes below weekly L3 + volume > 1.5x 20-period 6h volume MA + price below 1d EMA34
- Fixed position size 0.25 to limit fee churn and manage drawdown
- ATR-based trailing stop (2.0x ATR) to lock in profits
- Weekly Camarilla levels derived from prior week's OHLC (no look-ahead)
- Designed for very low trade frequency (target: 50-150 trades over 4 years) to avoid fee drag
- Works in bull markets (buying above weekly H3 with 1d EMA34 uptrend) and bear markets (selling below weekly L3 with 1d EMA34 downtrend)
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
    
    # Get 1d data for EMA34 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 1w data for weekly Camarilla pivot points (HTF)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla pivot points
    # P = (H+L+C)/3, range = H-L
    # H3 = P + 1.1*(H-L)/2, L3 = P - 1.1*(H-L)/2
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    h3_1w = pivot_1w + 1.1 * range_1w / 2.0
    l3_1w = pivot_1w - 1.1 * range_1w / 2.0
    
    # Align weekly Camarilla levels to 6h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1w, h3_1w)
    l3_aligned = align_htf_to_ltf(prices, df_1w, l3_1w)
    
    # Get 6h data for volume confirmation and ATR (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Volume average (20-period) on 6h for confirmation
    volume_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # ATR (10-period) on 6h for stoploss
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Align all indicators to 6h timeframe (primary)
    volume_ma_aligned = align_htf_to_ltf(prices, df_6h, volume_ma_20)
    atr_aligned = align_htf_to_ltf(prices, df_6h, atr_10)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    atr_stop = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(atr_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        h3_val = h3_aligned[i]
        l3_val = l3_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and 1d EMA34 trend filter
            # Long: price closes above weekly H3 + volume spike + price above 1d EMA34
            if price > h3_val and vol > 1.5 * vol_ma and price > ema_34_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
                atr_stop = entry_price - 2.0 * atr_val
            # Short: price closes below weekly L3 + volume spike + price below 1d EMA34
            elif price < l3_val and vol > 1.5 * vol_ma and price < ema_34_val:
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

name = "6h_WeeklyCamarilla_H3L3_1dEMA34_VolumeSpike_ATRTrail"
timeframe = "6h"
leverage = 1.0