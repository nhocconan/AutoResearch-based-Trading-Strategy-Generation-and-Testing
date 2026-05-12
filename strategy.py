#!/usr/bin/env python3
name = "1d_Camarilla_Pivot_Hybrid_v1"
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
    
    # 1W trend filter: EMA34 (long-term trend)
    df_1w = get_htf_data(prices, '1w')
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate daily Camarilla pivot levels from previous day
    # Using previous day's HLC
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    
    # First day will have NaN from roll, handle later
    range_prev = prev_high - prev_low
    camarilla_H4 = prev_close + range_prev * 1.1 / 2
    camarilla_H3 = prev_close + range_prev * 1.1 / 4
    camarilla_H2 = prev_close + range_prev * 1.1 / 6
    camarilla_H1 = prev_close + range_prev * 1.1 / 12
    camarilla_L1 = prev_close - range_prev * 1.1 / 12
    camarilla_L2 = prev_close - range_prev * 1.1 / 6
    camarilla_L3 = prev_close - range_prev * 1.1 / 4
    camarilla_L4 = prev_close - range_prev * 1.1 / 2
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Need enough data for 1w EMA34 and Camarilla (needs prev day)
    
    for i in range(start_idx, n):
        # Skip if 1w trend data not ready
        if np.isnan(ema34_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
            
        # Skip if Camarilla levels not ready (first day)
        if np.isnan(camarilla_H4[i]) or np.isnan(camarilla_L4[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price touches L3 or L4 + volume + 1w uptrend
            if ((low[i] <= camarilla_L3[i] or low[i] <= camarilla_L4[i]) and
                vol_filter[i] and
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price touches H3 or H4 + volume + 1w downtrend
            elif ((high[i] >= camarilla_H3[i] or high[i] >= camarilla_H4[i]) and
                  vol_filter[i] and
                  close[i] < ema34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reaches H3 (profit target) or closes below L1 (invalid)
            if (high[i] >= camarilla_H3[i] or close[i] < camarilla_L1[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches L3 (profit target) or closes above H1 (invalid)
            if (low[i] <= camarilla_L3[i] or close[i] > camarilla_H1[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals