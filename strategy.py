#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dTrend_VolumeConfirm
Hypothesis: Camarilla R3/S3 breakouts on 4h with 1d EMA50 trend filter and volume confirmation.
Only trade breakouts in direction of daily trend with volume > 1.5x 20-period average.
Uses discrete position sizing (0.25) to minimize fee churn. Target: 20-40 trades/year.
Designed to work in both bull and bear markets via trend alignment and volume confirmation.
Camarilla levels provide precise support/resistance; breaks with volume and trend are high-probability.
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF EMA50 to 4h timeframe (standard 1-bar delay for EMA)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d, additional_delay_bars=1)
    
    # Calculate 20-period volume average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema50_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Calculate Camarilla levels for current 4h bar
        # Using previous bar's high/low/close (standard Camarilla calculation)
        if i == 0:
            continue
        phigh = high[i-1]
        plow = low[i-1]
        pclose = close[i-1]
        rang = phigh - plow
        
        if rang <= 0:
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
            
        # Camarilla R3 and S3 levels
        r3 = pclose + rang * 1.1/4
        s3 = pclose - rang * 1.1/4
        
        if position == 0:
            # Look for breakout signals with trend and volume filters
            # Long: price breaks above R3 in uptrend (close > EMA50) with volume confirmation
            # Short: price breaks below S3 in downtrend (close < EMA50) with volume confirmation
            vol_confirm = volume[i] > 1.5 * vol_ma[i]
            long_signal = (close[i] > r3) and (close[i] > ema50_aligned[i]) and vol_confirm
            short_signal = (close[i] < s3) and (close[i] < ema50_aligned[i]) and vol_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below EMA50 (trend reversal) or breaks S3 (mean reversion)
            exit_signal = close[i] < ema50_aligned[i] or close[i] < s3
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above EMA50 (trend reversal) or breaks R3 (mean reversion)
            exit_signal = close[i] > ema50_aligned[i] or close[i] > r3
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dTrend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0