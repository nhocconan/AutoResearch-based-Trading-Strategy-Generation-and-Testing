#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter, volume confirmation (>1.3x 20-bar avg volume), and daily choppiness regime filter (CHOP < 38.2 = trend -> allow breakout entries). This strategy targets the proven winning pattern: tight price channel breakout + volume + regime filter. Uses 4h timeframe to target 75-200 total trades over 4 years. Daily trend and regime filters reduce false signals in bear markets like 2022 and 2025+. Discrete position sizing (0.25) minimizes fee churn.

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeChopRegime_v1"
timeframe = "4h"
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
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily Choppiness Index (CHOP) on 14-period for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d_arr, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d_arr, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    true_range_sum = pd.Series(true_range).rolling(window=14, min_periods=14).sum().values
    chop_1d = 100 * np.log10(atr_sum / true_range_sum) / np.log10(14)
    chop_1d = np.where(true_range_sum == 0, 50, chop_1d)  # avoid div by zero
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate Camarilla pivot levels from daily OHLC
    # R1 = close + (high - low) * 1.1 / 4
    # S1 = close - (high - low) * 1.1 / 4
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels to LTF (4h)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Close breaks above R1, price > 1d EMA34, volume spike (>1.3x avg), trending regime (CHOP < 38.2)
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 1.3 * avg_volume[i] and 
                chop_1d_aligned[i] < 38.2):
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below S1, price < 1d EMA34, volume spike (>1.3x avg), trending regime (CHOP < 38.2)
            elif (close[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 1.3 * avg_volume[i] and 
                  chop_1d_aligned[i] < 38.2):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close position if price drops below S1 (reversal) OR chop becomes too high (choppy market)
            if (close[i] < camarilla_s1_aligned[i] or 
                chop_1d_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close position if price rises above R1 (reversal) OR chop becomes too high (choppy market)
            if (close[i] > camarilla_r1_aligned[i] or 
                chop_1d_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals