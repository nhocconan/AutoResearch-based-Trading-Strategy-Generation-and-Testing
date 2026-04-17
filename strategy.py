#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 12h EMA trend filter + 1d Camarilla pivot breakout + volume confirmation.
Long when price breaks above daily Camarilla R3 level with 12h EMA34 > EMA89 (uptrend) and volume > 1.3x 20-period 4h volume average.
Short when price breaks below daily Camarilla S3 level with 12h EMA34 < EMA89 (downtrend) and volume > 1.3x 20-period 4h volume average.
Uses discrete position sizing 0.25 to limit fee drag. Target: 75-150 total trades over 4 years.
Daily pivots provide structural levels; 12h EMA crossover filters for trending markets only; volume confirms participation.
Designed to work in bull markets (breakout continuation) and bear markets (strong trend continuation).
"""

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
    
    # Get 4h data for volume MA
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Get 12h data for EMA trend
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 12h EMA34 and EMA89 for trend
    def ema(values, span):
        return pd.Series(values).ewm(span=span, adjust=False, min_periods=span).mean().values
    
    ema_34_12h = ema(close_12h, 34)
    ema_89_12h = ema(close_12h, 89)
    
    # Calculate 4h volume 20-period average
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate daily Camarilla levels
    # Camarilla: R3 = C + ((H-L) * 1.1/4), S3 = C - ((H-L) * 1.1/4)
    camarilla_r3 = close_1d + ((high_1d - low_1d) * 1.1 / 4)
    camarilla_s3 = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    
    # Align all to primary timeframe (4h)
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    ema_89_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_89_12h)
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for EMA89 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(ema_89_12h_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma_20_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * vol_ma_20_4h_aligned[i]
        # Trend filter: EMA34 > EMA89 for uptrend, EMA34 < EMA89 for downtrend
        uptrend = ema_34_12h_aligned[i] > ema_89_12h_aligned[i]
        downtrend = ema_34_12h_aligned[i] < ema_89_12h_aligned[i]
        
        if position == 0:
            # Long: price breaks above daily Camarilla R3 with uptrend and volume
            if (close[i] > camarilla_r3_aligned[i] and 
                uptrend and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below daily Camarilla S3 with downtrend and volume
            elif (close[i] < camarilla_s3_aligned[i] and 
                  downtrend and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below daily Camarilla R2 level
            camarilla_r2 = close_1d + ((high_1d - low_1d) * 1.1/6)
            camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
            if close[i] < camarilla_r2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above daily Camarilla S2 level
            camarilla_s2 = close_1d - ((high_1d - low_1d) * 1.1/6)
            camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
            if close[i] > camarilla_s2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12hEMA34_89_1dCamarilla_S3R3_Volume_Confirm"
timeframe = "4h"
leverage = 1.0