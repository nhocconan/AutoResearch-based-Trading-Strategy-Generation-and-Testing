#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_Volume_Control_v3
Hypothesis: Reduce trade frequency by tightening volume confirmation and adding ATR volatility filter.
Focus on high-probability breakouts at daily Camarilla R3/S3 levels with 1d trend filter, volume surge (3x),
and ATR-based volatility filter to avoid choppy markets. Targets 15-30 trades/year to minimize fee drag.
Designed to work in both bull and bear markets by requiring trend alignment and avoiding low-volatility environments.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla levels from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla R3 and S3 levels
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all higher timeframe data to 4h
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Trend filter: price > EMA50 = bullish, < EMA50 = bearish
    d1_uptrend = close > ema_50_1d_aligned
    d1_downtrend = close < ema_50_1d_aligned
    
    # Volume confirmation: current volume > 3.0x 20-period average (tighter)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 3.0)
    
    # Volatility filter: ATR > 0.5 * 50-period ATR average (avoid low volatility)
    atr_ma_50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ma_50_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50)
    volatility_filter = atr > (atr_ma_50_aligned * 0.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_surge[i]) or
            np.isnan(volatility_filter[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with trend alignment, volume surge, and volatility filter
        # Long: price breaks above R3 + daily uptrend + volume surge + volatility filter
        long_entry = (close[i] > R3_aligned[i] and 
                     d1_uptrend[i] and 
                     volume_surge[i] and
                     volatility_filter[i])
        
        # Short: price breaks below S3 + daily downtrend + volume surge + volatility filter
        short_entry = (close[i] < S3_aligned[i] and 
                      d1_downtrend[i] and 
                      volume_surge[i] and
                      volatility_filter[i])
        
        # Exit on opposite level break with volume surge
        long_exit = close[i] < S3_aligned[i] and volume_surge[i]
        short_exit = close[i] > R3_aligned[i] and volume_surge[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume_Control_v3"
timeframe = "4h"
leverage = 1.0