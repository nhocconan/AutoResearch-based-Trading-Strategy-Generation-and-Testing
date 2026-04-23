#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R extreme reversal with 1d EMA50 trend filter and volume confirmation
- Long when Williams %R crosses above -80 (oversold reversal) AND close > 1d EMA50 AND volume > 1.8x 20-period average
- Short when Williams %R crosses below -20 (overbought reversal) AND close < 1d EMA50 AND volume > 1.8x 20-period average
- Exit when Williams %R crosses -50 (mean reversion to midpoint)
- Uses 1d EMA50 for HTF trend alignment to avoid counter-trend entries
- Volume spike threshold set to 1.8x to reduce false signals
- Williams %R (14) captures short-term extremes in both bull and bear markets
- Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag
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
    
    # Get 1d data for EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Williams %R (14) on 4h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 14, 20)  # Need 50 for EMA50, 14 for Williams %R, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R conditions
        wr_above_oversold = williams_r[i] > -80  # Above oversold threshold
        wr_below_overbought = williams_r[i] < -20  # Below overbought threshold
        wr_crossed_above_oversold = williams_r[i] > -80 and williams_r[i-1] <= -80  # Cross above -80
        wr_crossed_below_overbought = williams_r[i] < -20 and williams_r[i-1] >= -20  # Cross below -20
        wr_crossed_above_mid = williams_r[i] > -50 and williams_r[i-1] <= -50  # Cross above -50
        wr_crossed_below_mid = williams_r[i] < -50 and williams_r[i-1] >= -50  # Cross below -50
        
        # Trend filter (using 1d EMA50)
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold reversal) + uptrend + volume confirmation
            if wr_crossed_above_oversold and uptrend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought reversal) + downtrend + volume confirmation
            elif wr_crossed_below_overbought and downtrend and volume_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R crosses -50 (mean reversion to midpoint)
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses above -50
                if wr_crossed_above_mid:
                    exit_signal = True
            elif position == -1:
                # Exit short: Williams %R crosses below -50
                if wr_crossed_below_mid:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsR_Extreme_1dEMA50_Trend_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0