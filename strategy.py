#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R4/S4 breakout with 1d volume spike and EMA trend filter.
# Uses Camarilla pivot levels (R4/S4) from prior 1d for stronger breakout structure,
# volume > 1.5x 20-bar EMA volume for conviction, and EMA50 > EMA200 on 4h for bull trend bias.
# Designed to capture strong breakouts in bull markets while avoiding false signals in ranging/bear conditions.
# Discrete position sizing (0.0, ±0.30) minimizes fee churn. Targets 20-40 trades/year per symbol.

name = "4h_Camarilla_R4S4_Breakout_1dVolumeSpike_EMATrend_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 4h Indicators (LTF) ---
    # EMA50 and EMA200 for trend filter
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    bull_trend = ema50 > ema200  # Only take longs in bull trend
    
    # Volume spike: volume > 1.5x 20-bar EMA volume
    volume_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ema20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels (R4, S4) from prior 1d bar
    camarilla_range = high_1d - low_1d
    r4_1d = close_1d + 1.5 * camarilla_range / 2.0
    s4_1d = close_1d - 1.5 * camarilla_range / 2.0
    
    # Align to 4h (wait for completed 1d bar)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(ema50[i]) or np.isnan(ema200[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Only trade long in bull trend (EMA50 > EMA200)
        if not bull_trend[i]:
            # In bear/range, stay flat or exit
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                # Exit long if price touches S4 (mean reversion)
                if close[i] <= s4_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.30
            elif position == -1:
                # Exit short if price touches R4 (mean reversion)
                if close[i] >= r4_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.30
            continue
        
        # Bull trend: look for breakouts with volume confirmation
        if position == 0:
            # LONG: Price breaks above R4 AND volume spike
            if close[i] > r4_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.30
                position = 1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S4 (mean reversion) OR volume drops
            if close[i] < s4_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
    
    return signals