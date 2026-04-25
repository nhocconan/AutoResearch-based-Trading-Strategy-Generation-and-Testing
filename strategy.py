#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dRegime
Hypothesis: On 6h timeframe, Elder Ray Bull Power and Bear Power combined with 1d regime filter (ADX) to capture trending moves while avoiding sideways chop.
Bull Power = High - EMA13, Bear Power = EMA13 - Low. Strong bullish when Bull Power rising and positive, strong bearish when Bear Power falling and negative.
1d ADX > 25 indicates trending regime where Elder Ray signals are reliable. Works in bull markets via Bull Power strength and bear markets via Bear Power strength.
Designed for 12-37 trades/year (50-150 over 4 years) to minimize fee drag.
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
    
    # 1d data for ADX regime filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX on 1d timeframe
    # +DM, -DM, TR
    up_move = df_1d['high'].diff()
    down_move = df_1d['low'].diff()
    up_move = up_move.where(up_move > down_move, 0.0)
    down_move = down_move.where(down_move > up_move, 0.0)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    plus_dm_smooth = up_move.ewm(alpha=alpha, adjust=False).mean()
    minus_dm_smooth = down_move.ewm(alpha=alpha, adjust=False).mean()
    tr_smooth = tr.ewm(alpha=alpha, adjust=False).mean()
    
    # DI+ and DI-
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    adx_values = adx.values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Calculate EMA13 on 6h for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA13 (13) and ADX calculation
    start_idx = 50  # enough for ADX smoothing
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        
        # Regime filter: only trade when ADX > 25 (trending market)
        if adx_aligned[i] < 25:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for entry signals
            # Long: Bull Power > 0 and rising (current > previous)
            # Short: Bear Power > 0 and rising (current > previous)
            if i > start_idx:
                prev_bull_power = bull_power[i-1]
                prev_bear_power = bear_power[i-1]
                
                long_entry = (curr_bull_power > 0) and (curr_bull_power > prev_bull_power)
                short_entry = (curr_bear_power > 0) and (curr_bear_power > prev_bear_power)
                
                if long_entry:
                    signals[i] = 0.25
                    position = 1
                elif short_entry:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when Bull Power becomes negative or starts falling
            if curr_bull_power <= 0 or curr_bull_power < bull_power[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when Bear Power becomes negative or starts falling
            if curr_bear_power <= 0 or curr_bear_power < bear_power[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_1dRegime"
timeframe = "6h"
leverage = 1.0