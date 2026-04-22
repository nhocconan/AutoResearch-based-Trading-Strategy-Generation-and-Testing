#!/usr/bin/env python3

"""
Hypothesis: 4-hour Bollinger Squeeze with 1-day ATR volatility filter and volume confirmation.
Trades breakouts from low-volatility periods in the direction of the daily ATR trend.
Uses Bollinger Band width to detect squeeze, ATR to confirm volatility regime, and volume to confirm breakout.
Designed for low trade frequency (20-50 trades/year) to minimize fee drag and work in both bull and bear markets
by trading mean-reversion breakouts during low volatility and trend-following during high volatility.
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
    
    # Load daily data for ATR filter and Bollinger calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily ATR for volatility regime filter (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Bollinger Bands (20, 2) - 4h close
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    bb_width = (bb_upper - bb_lower) / sma_20  # normalized width
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(sma_20[i]) or 
            np.isnan(std_20[i]) or np.isnan(bb_width[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Bollinger Squeeze: width below 20-period average of width
        bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean()
        bb_width_ma_val = bb_width_ma.iloc[i] if not np.isnan(bb_width_ma.iloc[i]) else 0
        squeeze = bb_width[i] < 0.5 * bb_width_ma_val  # tight squeeze
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0 and squeeze and vol_spike:
            # Long: breakout above upper band with rising volatility
            if close[i] > bb_upper[i] and atr_14_1d_aligned[i] > atr_14_1d_aligned[max(0, i-1)]:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below lower band with rising volatility
            elif close[i] < bb_lower[i] and atr_14_1d_aligned[i] > atr_14_1d_aligned[max(0, i-1)]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: volatility contraction or opposite band touch
            exit_signal = False
            
            if position == 1:
                # Exit long: price touches lower band or volatility contracts
                if close[i] < bb_lower[i] or atr_14_1d_aligned[i] < atr_14_1d_aligned[max(0, i-1)] * 0.8:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price touches upper band or volatility contracts
                if close[i] > bb_upper[i] or atr_14_1d_aligned[i] < atr_14_1d_aligned[max(0, i-1)] * 0.8:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Bollinger_Squeeze_1dATR_Volume"
timeframe = "4h"
leverage = 1.0