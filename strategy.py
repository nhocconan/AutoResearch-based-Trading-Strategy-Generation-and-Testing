#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with volume confirmation and Choppiness Index regime filter.
# Long when price breaks above upper Donchian band with volume spike and trending regime (CHOP < 38.2).
# Short when price breaks below lower Donchian band with volume spike and trending regime (CHOP < 38.2).
# Exit when price returns to the middle of the Donchian channel (mean reversion).
# Uses weekly trend filter (price > weekly SMA50 for longs, price < weekly SMA50 for shorts) to avoid counter-trend trades.
# Designed for 12h timeframe to capture medium-term trends with low frequency and high accuracy.
# Target: 20-40 trades/year per symbol to minimize fee drag.

name = "12h_Donchian_Volume_Chop_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Choppiness Index (14-period) - measures ranging vs trending markets
    # CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    atr_list = []
    for i in range(n):
        if i < 1:
            atr_list.append(0)
        else:
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            atr_list.append(tr)
    atr_series = pd.Series(atr_list)
    atr_ma = atr_series.rolling(window=14, min_periods=14).mean()
    
    # True Range for denominator
    tr_list = []
    for i in range(n):
        if i < 1:
            tr_list.append(high[i] - low[i])
        else:
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            tr_list.append(tr)
    tr_sum = pd.Series(tr_list).rolling(window=14, min_periods=14).sum()
    
    chop = 100 * np.log10(tr_sum / atr_ma) / np.log10(14)
    chop = chop.fillna(50).values  # neutral when undefined
    
    # Volume spike: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    # Weekly trend filter (price > weekly SMA50 for long bias, < for short bias)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    weekly_close = df_1w['close'].values
    weekly_sma50 = pd.Series(weekly_close).rolling(window=50, min_periods=50).mean().values
    weekly_sma50_aligned = align_htf_to_ltf(prices, df_1w, weekly_sma50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(chop[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(weekly_sma50_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above upper Donchian with volume and trending regime + weekly uptrend
            if (close[i] > donchian_high[i] and 
                volume_spike[i] and 
                chop[i] < 38.2 and 
                close[i] > weekly_sma50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian with volume and trending regime + weekly downtrend
            elif (close[i] < donchian_low[i] and 
                  volume_spike[i] and 
                  chop[i] < 38.2 and 
                  close[i] < weekly_sma50_aligned[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit when price returns to middle of Donchian channel (mean reversion)
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit when price returns to middle of Donchian channel (mean reversion)
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals