#!/usr/bin/env python3
"""
1d_Camarilla_H3L3_Breakout_1wEMA34_Trend_VolumeSpike
Hypothesis: Camarilla H3/L3 levels on 1d act as institutional support/resistance; price breaking these levels with volume spike and 1w EMA34 trend filter captures strong momentum moves in both bull and bear markets. Weekly EMA ensures we only trade with the higher timeframe trend, reducing false breaks in choppy conditions.
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
    
    # 1d data for Camarilla pivot levels and ATR (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Camarilla pivot levels on 1d (using previous day's OHLC)
    # H3 = close + 1.1*(high - low)/4
    # L3 = close - 1.1*(high - low)/4
    # We need previous day's data, so shift by 1
    if len(df_1d) < 2:
        return np.zeros(n)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 4
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 4
    # Align to 1d timeframe (already completed bars)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 1w EMA34 for trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    ema_34 = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Volume spike: current volume > 2.0 * 20-period volume MA (on 1d timeframe)
    # We need to compute volume MA on 1d and align
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    vol_spike = volume > (2.0 * vol_ma_20_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need 1d Camarilla (2 days for prev data) + 1w EMA (34) + volume MA (20)
    start_idx = max(2, 34, 20) + 5  # Conservative warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Long: price breaks above Camarilla H3 (resistance) with volume spike and 1w uptrend
            long_breakout = (curr_close > camarilla_h3_aligned[i]) and vol_spike[i] and (curr_close > ema_34_aligned[i])
            # Short: price breaks below Camarilla L3 (support) with volume spike and 1w downtrend
            short_breakout = (curr_close < camarilla_l3_aligned[i]) and vol_spike[i] and (curr_close < ema_34_aligned[i])
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below Camarilla L3 or trend turns down
            if (curr_close < camarilla_l3_aligned[i]) or (curr_close < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Camarilla H3 or trend turns up
            if (curr_close > camarilla_h3_aligned[i]) or (curr_close > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_H3L3_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0