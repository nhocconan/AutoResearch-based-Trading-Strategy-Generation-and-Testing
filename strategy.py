#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + volume confirmation + 1d choppiness regime filter
# In choppy markets (CHOP > 61.8): mean reversion at Donchian bands (fade extremes)
# In trending markets (CHOP < 38.2): breakout continuation (trade with trend)
# Uses 12h for entry timing and 1d for regime detection via Choppiness Index
# Position size 0.25 to limit drawdown and enable discrete levels
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Works in both bull/bear: adapts to regime via chop filter

name = "12h_1d_donchian_chop_regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Choppiness Index (CHOP)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr_1d = np.zeros(len(df_1d))
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr0 = high_1d[i] - low_1d[i]
        tr1 = abs(high_1d[i] - close_1d[i-1])
        tr2 = abs(low_1d[i] - close_1d[i-1])
        tr_1d[i] = max(tr0, tr1, tr2)
    
    # ATR(14) - sum of TR
    atr_1d = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        if i < 14:
            if i == 0:
                atr_1d[i] = tr_1d[i]
            else:
                atr_1d[i] = (atr_1d[i-1] * i + tr_1d[i]) / (i + 1)
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # Highest high and lowest low over 14 periods
    hh_1d = np.zeros(len(df_1d))
    ll_1d = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        if i < 14:
            hh_1d[i] = np.max(high_1d[:i+1])
            ll_1d[i] = np.min(low_1d[:i+1])
        else:
            hh_1d[i] = np.max(high_1d[i-13:i+1])
            ll_1d[i] = np.min(low_1d[i-13:i+1])
    
    # Chop calculation: 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    chop_1d = np.full(len(df_1d), 50.0)  # default to neutral
    for i in range(14, len(df_1d)):
        if hh_1d[i] != ll_1d[i]:
            chop_1d[i] = 100 * np.log10(atr_1d[i] / (hh_1d[i] - ll_1d[i])) / np.log10(14)
    
    # Align 1d chop to 12h timeframe
    chop_12h = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 12h Donchian channels (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            highest_high[i] = np.max(high[:i+1])
            lowest_low[i] = np.min(low[:i+1])
        else:
            highest_high[i] = np.max(high[i-19:i+1])
            lowest_low[i] = np.min(low[i-19:i+1])
    
    # Calculate 12h volume SMA(20) for confirmation
    volume_sma = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            volume_sma[i] = np.mean(volume[:i+1])
        else:
            volume_sma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(volume_sma[i]) or 
            np.isnan(chop_12h[i])):
            signals[i] = 0.0
            continue
        
        chop = chop_12h[i]
        vol = volume[i]
        vol_ma = volume_sma[i]
        
        if position == 1:  # Long position
            # Exit conditions
            if chop > 61.8:  # Choppy regime - mean reversion
                # Exit when price returns to middle of channel
                if close[i] <= (highest_high[i] + lowest_low[i]) / 2:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # Trending regime - breakout continuation
                # Exit when price breaks below lowest low (stoploss)
                if close[i] < lowest_low[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            # Exit conditions
            if chop > 61.8:  # Choppy regime - mean reversion
                # Exit when price returns to middle of channel
                if close[i] >= (highest_high[i] + lowest_low[i]) / 2:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # Trending regime - breakout continuation
                # Exit when price breaks above highest high (stoploss)
                if close[i] > highest_high[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            # Entry logic based on regime
            if chop > 61.8:  # Choppy regime - mean reversion
                # Go long when price touches lower band with volume confirmation
                # Go short when price touches upper band with volume confirmation
                if close[i] <= lowest_low[i] and vol > vol_ma * 1.5:
                    position = 1
                    signals[i] = 0.25
                elif close[i] >= highest_high[i] and vol > vol_ma * 1.5:
                    position = -1
                    signals[i] = -0.25
            else:  # Trending regime - breakout continuation
                # Go long when price breaks above highest high with volume
                # Go short when price breaks below lowest low with volume
                if close[i] > highest_high[i] and vol > vol_ma * 1.5:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < lowest_low[i] and vol > vol_ma * 1.5:
                    position = -1
                    signals[i] = -0.25
    
    return signals