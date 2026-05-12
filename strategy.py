#!/usr/bin/env python3
# 4h_Camarilla_R3S3_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot levels (R3/S3) from daily timeframe act as strong support/resistance.
# Breakout above R3 or below S3 with volume confirmation and aligned with 1-day EMA50 trend captures
# institutional breakout moves. Works in bull/bear markets by only trading in direction of higher timeframe trend.
# Uses discrete position sizing (0.25) to limit drawdown and control trade frequency (~25-35 trades/year).

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume"
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
    
    # Load 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (R3, S3) from previous day
    # R3 = close + 1.1 * (high - low) / 2
    # S3 = close - 1.1 * (high - low) / 2
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    # First day will have NaN due to roll, handled by min_periods in EMA
    
    R3 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 2
    S3 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 2
    
    # 1-day EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(close_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Get current values
        r3 = R3_aligned[i]
        s3 = S3_aligned[i]
        ema50 = ema50_1d_aligned[i]
        close_1d = close_1d_aligned[i]
        
        trend_up = close_1d > ema50
        trend_down = close_1d < ema50
        
        vol_confirm = volume_confirm[i]
        
        if position == 0:
            # LONG: Price breaks above R3 AND 1d uptrend AND volume confirmation
            if close[i] > r3 and trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 AND 1d downtrend AND volume confirmation
            elif close[i] < s3 and trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price returns below R3 OR trend weakens
            if close[i] < r3 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns above S3 OR trend weakens
            if close[i] > s3 or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals