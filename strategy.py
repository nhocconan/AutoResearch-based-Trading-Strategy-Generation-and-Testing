#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d EMA200 trend filter and volume spike confirmation
# Long when Williams %R(14) crosses above -80 (oversold) + price > 1d EMA200 + volume > 2.0x 20-period avg
# Short when Williams %R(14) crosses below -20 (overbought) + price < 1d EMA200 + volume > 2.0x 20-period avg
# Uses Williams %R for mean reversion entries in ranging markets, filtered by 1d EMA200 trend
# Volume confirmation ensures institutional participation. Designed for low trade frequency (12-25/year)
# Works in both bull and bear markets by requiring trend alignment and volume spikes

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d Indicators: EMA200 ===
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === Primary Timeframe Indicators: Williams %R(14) ===
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high_14 - close) / (highest_high_14 - lowest_low_14)) * -100
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 60
    
    for i in range(warmup, n):
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Williams %R crosses above -80 (oversold recovery)
        # 2. Price above 1d EMA200 (uptrend filter)
        # 3. Volume confirmation
        if (williams_r[i] > -80 and williams_r[i-1] <= -80) and (close[i] > ema_200_1d_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Williams %R crosses below -20 (overbought rejection)
        # 2. Price below 1d EMA200 (downtrend filter)
        # 3. Volume confirmation
        elif (williams_r[i] < -20 and williams_r[i-1] >= -20) and (close[i] < ema_200_1d_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_WilliamsR_EMA200_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0