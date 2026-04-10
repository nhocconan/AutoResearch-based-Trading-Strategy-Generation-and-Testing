#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d regime filter (ADX)
# - Bull Power = High - EMA13, Bear Power = EMA13 - Low
# - Long: Bull Power > 0 AND Bear Power < 0 AND 1d ADX > 25 (trending) AND 1d +DI > -DI
# - Short: Bear Power > 0 AND Bull Power < 0 AND 1d ADX > 25 AND 1d -DI > +DI
# - Exit: Power signals weaken (Bull Power <= 0 for long, Bear Power <= 0 for short) OR 1d ADX < 20 (range)
# - Position sizing: 0.25 discrete level
# - Targets ~15-30 trades/year on 6h timeframe. Elder Ray measures bull/bear strength relative to EMA,
#   ADX confirms trend strength, avoiding whipsaws in ranging markets. Works in bull/bear markets by
#   only taking strong directional moves when trend is confirmed.

name = "6h_1d_elder_ray_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Pre-compute HTF data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Calculate 1d ADX for regime filter
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    atr_1d = np.zeros_like(tr_1d)
    plus_dm_1d = np.zeros_like(plus_dm)
    minus_dm_1d = np.zeros_like(minus_dm)
    
    # Initialize first values
    atr_1d[0] = tr_1d[0]
    plus_dm_1d[0] = plus_dm[0]
    minus_dm_1d[0] = minus_dm[0]
    
    # Wilder smoothing
    for i in range(1, len(tr_1d)):
        atr_1d[i] = atr_1d[i-1] - (atr_1d[i-1] / 14) + tr_1d[i]
        plus_dm_1d[i] = plus_dm_1d[i-1] - (plus_dm_1d[i-1] / 14) + plus_dm[i]
        minus_dm_1d[i] = minus_dm_1d[i-1] - (minus_dm_1d[i-1] / 14) + minus_dm[i]
    
    # Calculate +DI and -DI
    plus_di_1d = np.where(atr_1d != 0, (plus_dm_1d / atr_1d) * 100, 0)
    minus_di_1d = np.where(atr_1d != 0, (minus_dm_1d / atr_1d) * 100, 0)
    
    # Calculate DX and ADX
    dx_1d = np.where((plus_di_1d + minus_di_1d) != 0, 
                     np.abs((plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)) * 100, 0)
    
    adx_1d = np.zeros_like(dx_1d)
    adx_1d[0] = dx_1d[0]
    for i in range(1, len(dx_1d)):
        adx_1d[i] = (adx_1d[i-1] * 13 + dx_1d[i]) / 14  # Wilder smoothing for ADX
    
    # Align HTF indicators to LTF
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    plus_di_1d_aligned = align_htf_to_ltf(prices, df_1d, plus_di_1d)
    minus_di_1d_aligned = align_htf_to_ltf(prices, df_1d, minus_di_1d)
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(plus_di_1d_aligned[i]) or 
            np.isnan(minus_di_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for entries
            # Long entry: Bull Power > 0 AND Bear Power < 0 AND ADX > 25 AND +DI > -DI
            if (bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0 and 
                adx_1d_aligned[i] > 25 and plus_di_1d_aligned[i] > minus_di_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Bear Power > 0 AND Bull Power < 0 AND ADX > 25 AND -DI > +DI
            elif (bear_power_aligned[i] > 0 and bull_power_aligned[i] < 0 and 
                  adx_1d_aligned[i] > 25 and minus_di_1d_aligned[i] > plus_di_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Power signals weaken OR ADX < 20 (range regime)
            if position == 1:  # Long position
                if (bull_power_aligned[i] <= 0 or adx_1d_aligned[i] < 20):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if (bear_power_aligned[i] <= 0 or adx_1d_aligned[i] < 20):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals