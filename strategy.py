#!/usr/bin/env python3
"""
1d_Williams_Alligator_1wTrend_Spinoff_v1
Hypothesis: Williams Alligator on 1d with 1w trend filter and ATR-based position sizing.
Trades only in direction of 1w EMA13 trend. Long when Alligator jaws < teeth < lips (bullish alignment),
short when jaws > teeth > lips (bearish alignment). Uses ATR(14) for volatility-adjusted stops.
Designed for low trade frequency (7-25/year) to minimize fee drag and work in both bull/bear markets.
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
    
    # 1w data for EMA13 trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    ema_13_1w = pd.Series(df_1w['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_13_1w)
    
    # Williams Alligator on 1d: SMAs of median price
    df_1d = get_htf_data(prices, '1d')
    median_price = (df_1d['high'].values + df_1d['low'].values) / 2.0
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values  # 13-period, shifted 8
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values   # 8-period, shifted 5
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values    # 5-period, shifted 3
    
    # Align Alligator lines to 1d timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # ATR(14) for volatility-based stoploss on 1d
    tr1 = pd.Series(high).diff().abs().values
    tr2 = pd.Series(low).diff().abs().values
    tr3 = np.abs(high[1:] - close[:-1])
    tr3 = np.concatenate([[np.nan], tr3])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_multiplier = 2.5
    
    # Start index: need enough for Alligator (13+8=21) and 1w EMA (13)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_13_1w_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Alligator alignment: bullish when jaw < teeth < lips
        bullish_alignment = jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i]
        # Alligator alignment: bearish when jaw > teeth > lips
        bearish_alignment = jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]
        
        # 1w trend filter
        uptrend_1w = curr_close > ema_13_1w_aligned[i]
        downtrend_1w = curr_close < ema_13_1w_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: bullish Alligator alignment + 1w uptrend
            if bullish_alignment and uptrend_1w:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: bearish Alligator alignment + 1w downtrend
            elif bearish_alignment and downtrend_1w:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit on ATR stop or Alligator bearish crossover
            if curr_close < entry_price - atr_multiplier * atr[i]:
                signals[i] = 0.0
                position = 0
            elif jaw_aligned[i] > teeth_aligned[i]:  # Alligator death cross (jaws > teeth)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on ATR stop or Alligator bullish crossover
            if curr_close > entry_price + atr_multiplier * atr[i]:
                signals[i] = 0.0
                position = 0
            elif jaw_aligned[i] < teeth_aligned[i]:  # Alligator golden cross (jaws < teeth)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Williams_Alligator_1wTrend_Spinoff_v1"
timeframe = "1d"
leverage = 1.0