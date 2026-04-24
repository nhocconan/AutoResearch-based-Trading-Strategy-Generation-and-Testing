#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 12h for execution, HTF: 1d for Camarilla pivot levels and EMA trend.
- Camarilla pivot levels: H3/L3 are strong intraday resistance/support levels derived from prior day's range.
- Entry: Long when price breaks above H3 with volume spike and 1d EMA34 uptrend.
         Short when price breaks below L3 with volume spike and 1d EMA34 downtrend.
- Exit: When price retouches the prior day's close (pivot point) or opposite signal.
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla pivot levels for 1d: based on prior day's OHLC
    # H3 = Close + 1.1*(High - Low)/4
    # L3 = Close - 1.1*(High - Low)/4
    # Pivot = (High + Low + Close)/3
    prior_high = df_1d['high'].shift(1).values
    prior_low = df_1d['low'].shift(1).values
    prior_close = df_1d['close'].shift(1).values
    
    # Avoid division by zero or invalid calculations
    hl_range = prior_high - prior_low
    valid_range = hl_range > 0
    
    camarilla_h3 = np.where(valid_range, prior_close + 1.1 * hl_range / 4, prior_close)
    camarilla_l3 = np.where(valid_range, prior_close - 1.1 * hl_range / 4, prior_close)
    camarilla_pivot = np.where(valid_range, (prior_high + prior_low + prior_close) / 3, prior_close)
    
    # Align 1d indicators to 12h
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 12h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 + volume MA (need at least 2 days for prior OHLC)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(ema_34_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for Camarilla breakout signals with volume spike and trend filter
            if volume_spike[i]:
                # Long: Price breaks above H3 with volume spike in uptrend
                if close[i] > camarilla_h3_aligned[i] and close[i-1] <= camarilla_h3_aligned[i-1] and close[i] > ema_34_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Price breaks below L3 with volume spike in downtrend
                elif close[i] < camarilla_l3_aligned[i] and close[i-1] >= camarilla_l3_aligned[i-1] and close[i] < ema_34_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Price retouches pivot point or opposite signal
            if close[i] <= camarilla_pivot_aligned[i] and close[i-1] > camarilla_pivot_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price retouches pivot point or opposite signal
            if close[i] >= camarilla_pivot_aligned[i] and close[i-1] < camarilla_pivot_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0