#!/usr/bin/env python3
# Hypothesis: 12h timeframe with weekly RSI mean reversion and daily volume confirmation.
# Uses weekly RSI(14) for overbought/oversold signals and daily volume spike for confirmation.
# Weekly RSI avoids short-term noise while capturing medium-term reversals.
# Daily volume filter ensures institutional participation.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "12h_WeeklyRSI14_DailyVolume_MeanReversion"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for RSI(14) calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate weekly RSI(14)
    close_1w = pd.Series(df_1w['close'].values)
    delta = close_1w.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_values = rsi_1w.values
    
    # Align weekly RSI to 12h timeframe (wait for weekly bar to close)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w_values)
    
    # Get daily data for volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily average volume (20-period)
    vol_1d = pd.Series(df_1d['volume'].values)
    avg_vol_1d = vol_1d.rolling(window=20, min_periods=20).mean().values
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # Volume spike: current volume > 2.0x daily average volume
    volume_spike = volume > (2.0 * avg_vol_1d_aligned)
    
    # RSI levels for mean reversion
    rsi_overbought = 70
    rsi_oversold = 30
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold + volume spike
            if rsi_1w_aligned[i] < rsi_oversold and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought + volume spike
            elif rsi_1w_aligned[i] > rsi_overbought and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral (50) or overbought
            if rsi_1w_aligned[i] >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral (50) or oversold
            if rsi_1w_aligned[i] <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals