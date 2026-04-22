#!/usr/bin/env python3
"""
Hypothesis: 1-day Bollinger Band squeeze breakout with 1-week trend and volume confirmation.
Long when price breaks above upper Bollinger Band during low volatility (squeeze) and 1w EMA50 rising.
Short when price breaks below lower Bollinger Band during squeeze and 1w EMA50 falling.
Exit when price returns to middle Bollinger Band or 1w EMA50 reverses.
Bollinger squeeze identifies low volatility breakout setups; 1w EMA provides higher-timeframe trend filter;
volume breakout confirms institutional participation. Designed for low trade frequency by requiring volatility contraction
followed by expansion with trend alignment. Works in both bull and bear markets by following the 1w trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands: 20-period SMA, 2 standard deviations
    bb_period = 20
    bb_std = 2.0
    
    sma_bb = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma_bb + (bb_std_dev * bb_std)
    lower_band = sma_bb - (bb_std_dev * bb_std)
    middle_band = sma_bb
    
    # Bollinger Band Width for squeeze detection: (Upper - Lower) / Middle
    bb_width = (upper_band - lower_band) / middle_band
    # Squeeze: BB Width below 20-period average (low volatility)
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < bb_width_ma
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume > 1.5 * vol_ma_20
    
    # Load 1-week data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 50-period EMA on 1w close for trend
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(sma_bb[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(bb_width[i]) or np.isnan(bb_width_ma[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper BB during squeeze with volume surge and 1w EMA50 rising
            if (close[i] > upper_band[i] and squeeze[i] and vol_surge[i] and 
                ema50_1w_aligned[i] > ema50_1w_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower BB during squeeze with volume surge and 1w EMA50 falling
            elif (close[i] < lower_band[i] and squeeze[i] and vol_surge[i] and 
                  ema50_1w_aligned[i] < ema50_1w_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to middle BB or 1w EMA50 reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: Price <= middle band or 1w EMA50 turns down
                if close[i] <= middle_band[i] or ema50_1w_aligned[i] < ema50_1w_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price >= middle band or 1w EMA50 turns up
                if close[i] >= middle_band[i] or ema50_1w_aligned[i] > ema50_1w_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_BollingerSqueeze_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0