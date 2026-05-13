#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter, volume spike (>1.5x 24-bar avg), and chop regime filter (CHOP(14) < 38.2 = trending). Uses discrete position sizing (0.25) to minimize fee churn. Designed for BTC/ETH robustness via confluence of price structure, daily trend, volume, and regime filters. Daily EMA34 ensures alignment with intermediate trend, reducing false breakouts in chop. Camarilla R3/S3 levels provide institutional breakout levels with built-in stop/reverse logic.

name = "12h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeChopRegime_v1"
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
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Choppiness Index (CHOP) on 14-period for regime filter
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
    
    # Calculate Camarilla levels from prior 1d (using OHLC of completed 1d bar)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), 
    #            S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    # We use R3/S3 as breakout levels
    # Since we're on 12h timeframe, we need to use prior completed 1d bar's OHLC
    # We'll calculate these on 1d data and align to 12h
    cam_R3_1d = close_1d + 1.1 * (high_1d - low_1d)
    cam_S3_1d = close_1d - 1.1 * (high_1d - low_1d)
    cam_R3_1d_aligned = align_htf_to_ltf(prices, df_1d, cam_R3_1d)
    cam_S3_1d_aligned = align_htf_to_ltf(prices, df_1d, cam_S3_1d)
    
    # Calculate average volume for confirmation (24-period = 12 days on 12h)
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):  # start after lookback
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or 
            np.isnan(cam_R3_1d_aligned[i]) or 
            np.isnan(cam_S3_1d_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3, price > 1d EMA34, volume spike (>1.5x avg), trending regime (CHOP < 38.2)
            if (close[i] > cam_R3_1d_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i] and 
                chop_1d_aligned[i] < 38.2):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3, price < 1d EMA34, volume spike (>1.5x avg), trending regime (CHOP < 38.2)
            elif (close[i] < cam_S3_1d_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i] and 
                  chop_1d_aligned[i] < 38.2):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close position if price drops below Camarilla S3 (reversal) OR chop becomes too high (choppy market)
            if (close[i] < cam_S3_1d_aligned[i] or 
                chop_1d_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close position if price rises above Camarilla R3 (reversal) OR chop becomes too high (choppy market)
            if (close[i] > cam_R3_1d_aligned[i] or 
                chop_1d_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals