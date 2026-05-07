#!/usr/bin/env python3
name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

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
    
    # 1d trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    trend_up = close > ema34_1d_aligned
    trend_down = close < ema34_1d_aligned
    
    # Volume spike filter (volume > 1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    # Camarilla pivot levels from 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_high = high_1d + (high_1d - low_1d) * 1.1 / 12  # R3 level
    camarilla_low = low_1d - (high_1d - low_1d) * 1.1 / 12   # S3 level
    
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 3  # ~12 hours
    
    start_idx = 20  # Ensure volume MA is valid
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(camarilla_high_aligned[i]) or 
            np.isnan(camarilla_low_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price breaks above R3 level AND 1d uptrend AND volume spike
            if close[i] > camarilla_high_aligned[i] and trend_up[i] and vol_spike[i]:
                signals[i] = 0.30
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below S3 level AND 1d downtrend AND volume spike
            elif close[i] < camarilla_low_aligned[i] and trend_down[i] and vol_spike[i]:
                signals[i] = -0.30
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price closes below camarilla low OR trend turns down
            if close[i] < camarilla_low_aligned[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: Price closes above camarilla high OR trend turns up
            if close[i] > camarilla_high_aligned[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: Camarilla R3/S3 breakout with 1d trend filter and volume spike captures
# institutional breakouts in both bull and bear markets. The Camarilla levels
# (R3/S3) act as magnet levels where price often accelerates after breaking through.
# Volume spike confirms institutional participation. 1d trend filter ensures we
# trade in the direction of the higher timeframe trend. Cooldown prevents overtrading.
# Position size 0.30 balances risk and return. Works in bull markets (breaks above R3
# in uptrend) and bear markets (breaks below S3 in downtrend). Target: 20-50 trades/year.