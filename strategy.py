#!/usr/bin/env python3
"""
6h_Camarilla_R4_S4_Breakout_VolumeFilter
Hypothesis: 6h Camarilla R4/S4 breakouts with volume confirmation and 1d trend filter capture strong momentum moves.
Long when price breaks above R4 with volume > 1.5x average and 1d close > 1d EMA50.
Short when price breaks below S4 with volume > 1.5x average and 1d close < 1d EMA50.
Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year.
Works in bull/bear markets: breakouts capture strong moves, volume filter avoids false breakouts, HTF trend aligns with higher timeframe momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 6h Indicators (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Camarilla pivot levels (based on previous 6h bar)
    # R4 = close + 1.5 * (high - low)
    # S4 = close - 1.5 * (high - low)
    # Using previous bar's high/low/close to avoid look-ahead
    camarilla_r4 = np.roll(close_6h, 1) + 1.5 * (np.roll(high_6h, 1) - np.roll(low_6h, 1))
    camarilla_s4 = np.roll(close_6h, 1) - 1.5 * (np.roll(high_6h, 1) - np.roll(low_6h, 1))
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume_6h > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r4[i]) or np.isnan(camarilla_s4[i]) 
            or np.isnan(volume_filter[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        
        if position == 0:
            # Long: price breaks above R4 + volume confirmation + 1d uptrend
            if price > camarilla_r4[i] and volume_filter[i] and price > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S4 + volume confirmation + 1d downtrend
            elif price < camarilla_s4[i] and volume_filter[i] and price < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit: price returns to midline (previous bar close) or volume dries up
            if price <= np.roll(close_6h, 1)[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns to midline (previous bar close) or volume dries up
            if price >= np.roll(close_6h, 1)[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R4_S4_Breakout_VolumeFilter"
timeframe = "6h"
leverage = 1.0