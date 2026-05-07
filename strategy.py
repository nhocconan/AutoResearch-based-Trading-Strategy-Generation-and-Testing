#!/usr/bin/env python3
name = "6h_Liquidity_Zone_Reversal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Identify liquidity zones: previous day high/low and overnight range
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    overnight_high = df_1d['high'].values  # Current day high (forms during session)
    overnight_low = df_1d['low'].values    # Current day low
    
    # Liquidity zones: where stops are likely placed
    liquidity_high = np.maximum(prev_day_high, overnight_high)
    liquidity_low = np.minimum(prev_day_low, overnight_low)
    
    # Align liquidity zones to 6h timeframe
    liq_high_aligned = align_htf_to_ltf(prices, df_1d, liquidity_high)
    liq_low_aligned = align_htf_to_ltf(prices, df_1d, liquidity_low)
    
    # Daily trend filter: EMA(50) on daily close
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current volume > 1.5x 24-period average (1 day of 6h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)
    
    for i in range(start_idx, n):
        if (np.isnan(liq_high_aligned[i]) or np.isnan(liq_low_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_24[i] * 1.5
        
        if position == 0:
            # Long: rejection of liquidity low with volume in daily uptrend
            liq_reject_low = low[i] <= liq_low_aligned[i] * 1.001 and close[i] > liq_low_aligned[i]
            uptrend = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            
            if liq_reject_low and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: rejection of liquidity high with volume in daily downtrend
            elif high[i] >= liq_high_aligned[i] * 0.999 and close[i] < liq_high_aligned[i]:
                liq_reject_high = True
            else:
                liq_reject_high = False
                
            if liq_reject_high and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price reaches liquidity high or loses volume/momentum
            if (high[i] >= liq_high_aligned[i] * 0.999 or 
                volume[i] < vol_ma_24[i] or
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price reaches liquidity low or loses volume/momentum
            if (low[i] <= liq_low_aligned[i] * 1.001 or 
                volume[i] < vol_ma_24[i] or
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h liquidity zone reversal with daily trend and volume confirmation
# - Liquidity zones (prev day high/low and overnight range) represent stop clusters
# - Price rejection of these zones with volume indicates institutional absorption
# - Long when price rejects liquidity low in daily uptrend (smart money buying dips)
# - Short when price rejects liquidity high in daily downtrend (smart money selling rallies)
# - Volume confirmation (1.5x average) ensures genuine interest, not fakeouts
# - Works in bull (buy liquidity sweeps in uptrend) and bear (sell liquidity sweeps in downtrend)
# - Exit when price reaches opposite liquidity zone or momentum fades
# - Target: 15-35 trades/year, avoiding excessive frequency and fee drag
# - Effective in ranging markets where stops accumulate and in trending markets with pullbacks