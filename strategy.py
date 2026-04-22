#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for Choppiness Index calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for daily data
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Set first day TR to high-low since no previous close
    tr[0] = high_1d[0] - low_1d[0]
    
    # Calculate ATR(14) for daily data
    atr_period = 14
    atr = np.zeros(len(tr))
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Calculate Choppiness Index (14-period)
    chop_period = 14
    sum_tr = np.zeros(len(tr))
    sum_tr[chop_period-1] = np.sum(tr[:chop_period])
    for i in range(chop_period, len(tr)):
        sum_tr[i] = sum_tr[i-1] - tr[i-chop_period] + tr[i]
    
    highest_high = np.zeros(len(high_1d))
    lowest_low = np.zeros(len(low_1d))
    highest_high[chop_period-1] = np.max(high_1d[:chop_period])
    lowest_low[chop_period-1] = np.min(low_1d[:chop_period])
    for i in range(chop_period, len(high_1d)):
        highest_high[i] = max(highest_high[i-1], high_1d[i])
        lowest_low[i] = min(lowest_low[i-1], low_1d[i])
    
    # Choppiness Index formula: 100 * log10(sumTR/(ATR*period)) / log10(period)
    chop = np.zeros(len(tr))
    for i in range(chop_period-1, len(tr)):
        if sum_tr[i] > 0 and atr[i] > 0:
            chop[i] = 100 * np.log10(sum_tr[i] / (atr[i] * chop_period)) / np.log10(chop_period)
        else:
            chop[i] = 50.0  # neutral value
    
    # Choppiness regime: >61.8 = ranging (mean revert), <38.2 = trending
    chop_high = 61.8
    chop_low = 38.2
    
    # Calculate ATR(14) for 12h data for volatility filtering
    tr_12h1 = high - low
    tr_12h2 = np.abs(high - np.roll(close, 1))
    tr_12h3 = np.abs(low - np.roll(close, 1))
    tr_12h = np.maximum(tr_12h1, np.maximum(tr_12h2, tr_12h3))
    tr_12h[0] = high[0] - low[0]
    
    atr_12h_period = 14
    atr_12h = np.zeros(len(tr_12h))
    if len(tr_12h) >= atr_12h_period:
        atr_12h[atr_12h_period-1] = np.mean(tr_12h[:atr_12h_period])
        for i in range(atr_12h_period, len(tr_12h)):
            atr_12h[i] = (atr_12h[i-1] * (atr_12h_period - 1) + tr_12h[i]) / atr_12h_period
    
    # Volatility filter: only trade when volatility is above average
    vol_ma50 = pd.Series(atr_12h).rolling(window=50, min_periods=50).mean().values
    vol_filter = atr_12h > vol_ma50  # Trade when volatility is above 50-period average
    
    # Volume spike filter (20-period on 12h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20  # Require 1.5x volume for confirmation
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align Choppiness Index to 12-hour timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):  # Start after warmup
        # Skip if data not ready or outside session
        if (np.isnan(chop_aligned[i]) or
            np.isnan(vol_ma20[i]) or np.isnan(vol_ma50[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Ranging market (high chop) + price near support + volume spike
            if (chop_aligned[i] > chop_high and 
                close[i] <= low[i] * 1.02 and  # Near daily low
                vol_spike[i] and vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Ranging market (high chop) + price near resistance + volume spike
            elif (chop_aligned[i] > chop_high and 
                  close[i] >= high[i] * 0.98 and  # Near daily high
                  vol_spike[i] and vol_filter[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Market starts trending (low chop) OR opposite extreme touched
            if position == 1:
                if (chop_aligned[i] < chop_low or  # Market started trending
                    close[i] >= high[i] * 0.99):   # Hit resistance
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (chop_aligned[i] < chop_low or  # Market started trending
                    close[i] <= low[i] * 1.01):    # Hit support
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Chop_Range_MeanReversion_Vol_Session"
timeframe = "12h"
leverage = 1.0