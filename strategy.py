#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ChopFilter_v4
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA34 trend filter, volume spike confirmation, and choppiness regime filter. 
Enters long when price breaks above R1 with volume > 1.5x average and chop < 61.8 (trending), short when breaks below S1 with volume confirmation and chop < 61.8. 
Uses discrete position sizing (0.25) to minimize fee churn. Targets 50-150 trades over 4 years by requiring multiple confirmations.
Works in bull/bear via trend alignment: only takes longs in uptrend (price > EMA34), shorts in downtrend (price < EMA34).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend and chop filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ATR for volatility (used in chop and volume thresholds)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum([tr1, tr2, tr3], axis=0)
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    # Calculate 1d EMA34 for HTF trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    htf_trend = np.where(close > ema_34_1d_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate 1d choppiness index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (n * log10(highest_high - lowest_low))) / log10(n)
    chop_period = 14
    highest_high = pd.Series(df_1d['high'].values).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low = pd.Series(df_1d['low'].values).rolling(window=chop_period, min_periods=chop_period).min().values
    atr_1d = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values  # same ATR
    sum_atr_1d = pd.Series(atr_1d).rolling(window=chop_period, min_periods=chop_period).sum().values
    highest_lowest_diff = highest_high - lowest_low
    # Avoid division by zero
    highest_lowest_diff = np.where(highest_lowest_diff == 0, 1e-10, highest_lowest_diff)
    chop_1d = 100 * np.log10(sum_atr_1d / (chop_period * np.log10(highest_lowest_diff))) / np.log10(chop_period)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: R4 = close + 1.5*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4, etc.
    # We only need R1 and S1 for breakout
    # R1 = close + 1.1*(high-low)*1.1/12, S1 = close - 1.1*(high-low)*1.1/12
    # Using previous day's OHLC
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_range = prev_high - prev_low
    
    r1 = prev_close + 1.1 * prev_range * 1.1 / 12
    s1 = prev_close - 1.1 * prev_range * 1.1 / 12
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: current volume > 1.5x average volume
    vol_ma_period = 20
    vol_ma = pd.Series(volume).rolling(window=vol_ma_period, min_periods=vol_ma_period).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 14 for ATR/CHOP, 20 for volume MA)
    start_idx = max(34, chop_period, vol_ma_period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Only trade in trending regime (CHOP < 61.8)
        if chop_1d_aligned[i] < 61.8:
            # Long breakout above R1 with volume spike and uptrend HTF
            if close[i] > r1_aligned[i] and volume_spike[i] and htf_trend[i] == 1:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            # Short breakout below S1 with volume spike and downtrend HTF
            elif close[i] < s1_aligned[i] and volume_spike[i] and htf_trend[i] == -1:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            else:
                # Exit conditions: close position when price returns to midpoint or opposite level
                midpoint = (r1_aligned[i] + s1_aligned[i]) / 2
                if position == 1 and close[i] < midpoint:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close[i] > midpoint:
                    signals[i] = 0.0
                    position = 0
                else:
                    # Hold current position
                    if position == 0:
                        signals[i] = 0.0
                    elif position == 1:
                        signals[i] = 0.25
                    else:
                        signals[i] = -0.25
        else:
            # In choppy regime (CHOP >= 61.8), stay flat
            signals[i] = 0.0
            position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ChopFilter_v4"
timeframe = "12h"
leverage = 1.0