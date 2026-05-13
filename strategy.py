#!/usr/bin/env python3
"""
1d_WeeklyPivot_SupportResistance_Trend_Filter
Hypothesis: Weekly pivot points (support/resistance) act as strong daily-level barriers. 
Price breaking above weekly R1 with bullish weekly trend and volume confirmation = long.
Price breaking below weekly S1 with bearish weekly trend and volume confirmation = short.
Exit on return to weekly pivot (PP) or trend reversal. Uses weekly trend filter for higher timeframe bias.
Target: 10-25 trades/year per symbol to minimize fee decay in ranging markets.
"""

name = "1d_WeeklyPivot_SupportResistance_Trend_Filter"
timeframe = "1d"
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
    
    # Weekly pivot points (calculated from prior week)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points: PP = (H+L+C)/3, R1 = 2*PP - L, S1 = 2*PP - H
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    
    # Align weekly pivot levels to daily timeframe (available after weekly bar closes)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Weekly trend filter: EMA50 on weekly close
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = weekly_close > weekly_ema50
    weekly_downtrend = weekly_close < weekly_ema50
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend)
    
    # Volume confirmation: volume > 1.5 * 20-day average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        pp_val = pp_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        weekly_uptrend = weekly_uptrend_aligned[i]
        weekly_downtrend = weekly_downtrend_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: break above weekly R1, weekly uptrend, volume confirmation
            if close[i] > r1_val and weekly_uptrend and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: break below weekly S1, weekly downtrend, volume confirmation
            elif close[i] < s1_val and weekly_downtrend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: return to weekly pivot (PP) or weekly trend turns down
            if close[i] <= pp_val or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: return to weekly pivot (PP) or weekly trend turns up
            if close[i] >= pp_val or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals