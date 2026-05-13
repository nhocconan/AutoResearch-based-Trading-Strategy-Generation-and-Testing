#!/usr/bin/env python3
"""
12h_Williams_Vix_Fix_Long_Only
Hypothesis: Williams Vix Fix identifies market fear/spikes in volatility, signaling potential reversal points. Combined with 1d trend filter and volume confirmation, it captures mean-reversion bounces in oversold conditions during both bull and bear markets. Long-only bias reduces whipsaw in strong trends, focusing on high-probability bounce setups.
"""

name = "12h_Williams_Vix_Fix_Long_Only"
timeframe = "12h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and Williams Vix Fix calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 50-period EMA on 1d close for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams Vix Fix: measures market fear, higher = more fear
    # WVF = ((Highest Close in Period - Low) / Highest Close in Period) * 100
    # We use 22-period lookback (approx 1 month of trading days)
    highest_close = pd.Series(df_1d['close']).rolling(window=22, min_periods=22).max().values
    wvf = ((highest_close - df_1d['low'].values) / highest_close) * 100
    wvf_aligned = align_htf_to_ltf(prices, df_1d, wvf)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    for i in range(50, n):
        if position == 0:
            # LONG: High fear (WVF > 80) + volume spike + above 1d EMA50 (uptrend filter)
            if (wvf_aligned[i] > 80 and 
                volume_spike[i] and 
                close[i] > trend_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Fear subsides (WVF < 40) or trend turns down
            if (wvf_aligned[i] < 40 or 
                close[i] < trend_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals