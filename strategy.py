# 4h_Camarilla_R1_S1_Breakout_12hTrend_Volume_v2
# Hypothesis: Trade breakouts at R1/S1 levels from 12h timeframe, aligned with 1d trend, filtered by volume spike and chop regime to avoid whipsaws.
# Uses 12h for pivot calculation (structure) and 1d for trend (higher timeframe bias). Volume spike confirms institutional interest.
# Chop filter avoids ranging markets where breakouts fail. Works in bull (breakouts above R1 in uptrend) and bear (breakdowns below S1 in downtrend).
# Target: 25-40 trades/year per symbol to stay well under fee drag limits.

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_Volume_v2"
timeframe = "4h"
leverage = 1.0

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

    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Camarilla pivot levels (R1, S1) from previous 12h bar
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    camarilla_width = (high_12h - low_12h) * 1.1 / 12
    r1 = close_12h + camarilla_width
    s1 = close_12h - camarilla_width
    
    # 12h trend: EMA50
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1d data for chop filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Chopping index (14-period)
    def choppiness_index(high, low, close, period=14):
        atr = np.zeros(len(close))
        tr = np.zeros(len(close))
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        for i in range(period, len(close)):
            atr[i] = np.mean(tr[i-period+1:i+1])
        # Avoid division by zero
        atr_sum = np.nansum(atr[period-1:]) if np.any(~np.isnan(atr[period-1:])) else 1e-10
        if atr_sum == 0:
            atr_sum = 1e-10
        highest_high = np.max(high[period-1:]) if len(high[period-1:]) > 0 else close[-1]
        lowest_low = np.min(low[period-1:]) if len(low[period-1:]) > 0 else close[-1]
        range_max_min = highest_high - lowest_low
        if range_max_min == 0:
            range_max_min = 1e-10
        chop = 100 * np.log10(atr_sum / range_max_min) / np.log10(period)
        # Align to array length
        chop_full = np.full(len(close), np.nan)
        chop_full[period-1:] = chop
        return chop_full

    chop_1d = choppiness_index(high_1d, low_1d, close_1d, 14)
    
    # Align 12h indicators to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume spike: volume > 2.0 * 3-period average (1.5 days worth at 4h)
    vol_ma_3 = pd.Series(volume).rolling(window=3, min_periods=3).mean().values
    volume_spike = volume > 2.0 * vol_ma_3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or
            np.isnan(chop_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > R1 + 12h uptrend + volume spike + chop > 61.8 (ranging)
            if (close[i] > r1_aligned[i] and 
                close[i] > ema50_12h_aligned[i] and 
                volume_spike[i] and 
                chop_1d_aligned[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # SHORT: Close < S1 + 12h downtrend + volume spike + chop > 61.8 (ranging)
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema50_12h_aligned[i] and 
                  volume_spike[i] and 
                  chop_1d_aligned[i] > 61.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below S1 or trend reversal or chop < 38.2 (trending - exit range play)
            if (close[i] < s1_aligned[i] or 
                close[i] < ema50_12h_aligned[i] or
                chop_1d_aligned[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above R1 or trend reversal or chop < 38.2 (trending - exit range play)
            if (close[i] > r1_aligned[i] or 
                close[i] > ema50_12h_aligned[i] or
                chop_1d_aligned[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals