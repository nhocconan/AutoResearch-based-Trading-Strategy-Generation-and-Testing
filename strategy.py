#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dATR_VolumeSpike_v2
Hypothesis: Refined Camarilla R1/S1 breakout with 1d ATR volatility filter and volume spike confirmation.
Adds 4h EMA50 trend filter to reduce false breakouts and improve trade quality.
Target: 15-25 trades/year via tighter confluence of four filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivots and ATR (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla R1 and S1 levels (R1 = C + 1.1*(HL/4), S1 = C - 1.1*(HL/4))
    R1 = prev_close + 1.1 * prev_range * (1.0/4.0)
    S1 = prev_close - 1.1 * prev_range * (1.0/4.0)
    
    # Align 1d levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # 1d ATR(14) for volatility filter
    tr1 = np.maximum(df_1d['high'].values, np.roll(df_1d['close'].values, 1))
    tr1 = np.maximum(tr1, np.roll(df_1d['low'].values, 1))
    tr2 = np.abs(np.roll(df_1d['close'].values, 1) - df_1d['low'].values)
    tr3 = np.abs(np.roll(df_1d['close'].values, 1) - df_1d['high'].values)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan  # first value has no previous close
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Volume spike: current volume > 2.0 * 20-period volume MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    # 4h EMA50 for trend filter (calculated on 4h data)
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d previous data (1) + 1d ATR (14) + volume MA (20) + EMA50 (50)
    start_idx = max(14, 20, 50) + 1  # Conservative warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(ema_50[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volatility filter: avoid low-volatility chop (ATR < 0.5 * 20-period ATR MA)
        atr_ma_20 = pd.Series(atr_14_aligned).rolling(window=20, min_periods=20).mean().values
        low_vol_filter = atr_14_aligned[i] > (0.5 * atr_ma_20[i])
        
        # Trend filter: price above EMA50 for long, below for short
        uptrend = curr_close > ema_50[i]
        downtrend = curr_close < ema_50[i]
        
        if position == 0:
            # Look for entry signals with all filters
            # Long breakout: price breaks above R1 with volume spike, adequate volatility, and uptrend
            long_breakout = (curr_close > R1_aligned[i]) and vol_spike[i] and low_vol_filter and uptrend
            # Short breakout: price breaks below S1 with volume spike, adequate volatility, and downtrend
            short_breakout = (curr_close < S1_aligned[i]) and vol_spike[i] and low_vol_filter and downtrend
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit if price breaks below S1 (mean reversion) or volatility collapses or trend breaks
            if curr_close < S1_aligned[i] or not low_vol_filter or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit if price breaks above R1 (mean reversion) or volatility collapses or trend breaks
            if curr_close > R1_aligned[i] or not low_vol_filter or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dATR_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0