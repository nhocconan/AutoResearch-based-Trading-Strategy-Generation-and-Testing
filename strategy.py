#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Volume_Regime"
timeframe = "4h"
leverage = 1.0

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
    
    # Load daily data for Camarilla levels and EMA
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla R1 and S1 from previous day
    # R1 = close + (high - low) * 1.12
    # S1 = close - (high - low) * 1.12
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.12
    
    # Align to 4h (previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_avg)
    
    # Chop filter: avoid choppy markets (CHOP > 61.8)
    # Calculate ATR(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate max/min close over last 14 periods
    max_close = pd.Series(close).rolling(window=14, min_periods=14).max().values
    min_close = pd.Series(close).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(atr/14) / (max_close - min_close))
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    range_close = max_close - min_close
    chop = np.where(range_close > 0, 100 * np.log10(sum_atr / 14 / range_close), 50)
    chop_filter = chop <= 61.8  # Trending market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_filter[i]) or
            np.isnan(chop_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above R1 + above daily EMA34 + volume spike + trending
            if close[i] > camarilla_r1_aligned[i] and close[i] > ema_34_1d_aligned[i] and vol_filter[i] and chop_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below S1 + below daily EMA34 + volume spike + trending
            elif close[i] < camarilla_s1_aligned[i] and close[i] < ema_34_1d_aligned[i] and vol_filter[i] and chop_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price below S1 or below daily EMA34
            if close[i] < camarilla_s1_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above R1 or above daily EMA34
            if close[i] > camarilla_r1_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals