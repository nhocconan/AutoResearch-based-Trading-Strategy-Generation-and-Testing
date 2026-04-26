#!/usr/bin/env python3
"""
12h_WilliamsAlligator_JawTeethLips_1wTrend_VolumeFilter_v1
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) on 12h identifies trend direction and alignment. 
Only trade when all three lines are aligned (Jaw > Teeth > Lips for uptrend, reverse for downtrend).
Filter with 1w EMA34 trend and volume spike to avoid whipsaws. Exit on Alligator lines crossing or ATR stop.
Designed for low trade frequency (12-37/year) to minimize fee drag. Works in both bull/bear markets 
by using trend-following with volatility-adjusted exits.
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
    
    # Calculate Williams Alligator on 12h: Jaw (13,8), Teeth (8,5), Lips (5,3)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Jaw: 13-period SMMA, shifted 8 bars
    jaw_12h = pd.Series(close_12h).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMMA, shifted 5 bars
    teeth_12h = pd.Series(close_12h).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMMA, shifted 3 bars
    lips_12h = pd.Series(close_12h).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align to 12h timeframe (primary)
    jaw_12h_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_12h_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_12h_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    
    # 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for all indicators
    start_idx = max(20, 20, 14, 13, 34)  # volume avg, ATR, Alligator, 1w EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_12h_aligned[i]) or np.isnan(teeth_12h_aligned[i]) or np.isnan(lips_12h_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        atr_val = atr[i]
        size = 0.25  # 25% position size to manage risk
        
        if position == 0:
            # Flat - look for Alligator alignment with trend and volume confirmation
            # Long: Jaw > Teeth > Lips (bullish alignment) + price above 1w EMA34 + volume spike
            long_entry = (jaw_12h_aligned[i] > teeth_12h_aligned[i]) and \
                       (teeth_12h_aligned[i] > lips_12h_aligned[i]) and \
                       (close_val > ema_34_1w_aligned[i]) and \
                       volume_spike[i]
            # Short: Jaw < Teeth < Lips (bearish alignment) + price below 1w EMA34 + volume spike
            short_entry = (jaw_12h_aligned[i] < teeth_12h_aligned[i]) and \
                        (teeth_12h_aligned[i] < lips_12h_aligned[i]) and \
                       (close_val < ema_34_1w_aligned[i]) and \
                       volume_spike[i]
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on Alligator lines crossing (Teeth < Lips) or ATR stoploss
            exit_condition = (teeth_12h_aligned[i] < lips_12h_aligned[i]) or \
                           (close_val < entry_price - 2.5 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on Alligator lines crossing (Teeth > Lips) or ATR stoploss
            exit_condition = (teeth_12h_aligned[i] > lips_12h_aligned[i]) or \
                           (close_val > entry_price + 2.5 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "12h_WilliamsAlligator_JawTeethLips_1wTrend_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0