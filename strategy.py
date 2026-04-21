#!/usr/bin/env python3
"""
1d Weekly Keltner Channel Breakout with Volume Confirmation and ATR Stop
Hypothesis: Weekly Keltner Channels (based on ATR) provide dynamic support/resistance that adapts to volatility. Breakouts above upper channel or below lower channel with volume confirmation capture institutional moves, while ATR-based stops limit losses. Works in both bull and bear markets by using adaptive volatility-based channels and avoiding overtrading through strict weekly confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for Keltner Channel calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly True Range for ATR
    tr1 = np.abs(high_weekly - low_weekly)
    tr2 = np.abs(high_weekly - np.roll(close_weekly, 1))
    tr3 = np.abs(low_weekly - np.roll(close_weekly, 1))
    tr1[0] = high_weekly[0] - low_weekly[0]
    tr2[0] = np.abs(high_weekly[0] - close_weekly[0])
    tr3[0] = np.abs(low_weekly[0] - close_weekly[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_weekly = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate weekly EMA (20-period) as middle line
    ema_weekly = pd.Series(close_weekly).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Calculate Keltner Channels: Upper = EMA + 2*ATR, Lower = EMA - 2*ATR
    upper_keltner = ema_weekly + 2.0 * atr_weekly
    lower_keltner = ema_weekly - 2.0 * atr_weekly
    
    # Align weekly indicators to daily timeframe
    atr_weekly_aligned = align_htf_to_ltf(prices, df_weekly, atr_weekly)
    upper_keltner_aligned = align_htf_to_ltf(prices, df_weekly, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_weekly, lower_keltner)
    
    # Main timeframe data (daily)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(atr_weekly_aligned[i]) or np.isnan(upper_keltner_aligned[i]) or np.isnan(lower_keltner_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        atr = atr_weekly_aligned[i]
        upper = upper_keltner_aligned[i]
        lower = lower_keltner_aligned[i]
        vol_current = volume[i]
        
        # Volume filter: current volume > 1.5x 20-period average (selective but not too strict)
        vol_ma = np.mean(volume[max(0, i-20):i]) if i >= 20 else volume[i]
        vol_ok = vol_current > 1.5 * vol_ma
        
        if position == 0:
            # Long breakout: price breaks above upper Keltner channel with volume confirmation
            if price > upper and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below lower Keltner channel with volume confirmation
            elif price < lower and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below lower Keltner channel (failed breakout) or ATR-based stop
            if price < lower or (i > 0 and close[i-1] > lower and price < close[i-1] - 1.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above upper Keltner channel (failed breakdown) or ATR-based stop
            if price > upper or (i > 0 and close[i-1] < upper and price > close[i-1] + 1.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyKeltnerBreakout_Volume_ATRFilter"
timeframe = "1d"
leverage = 1.0