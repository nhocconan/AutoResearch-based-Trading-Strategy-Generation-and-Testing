#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot R1/S1 breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla pivot levels provide high-probability intraday support/resistance.
# Breakout above R1 or below S1 captures momentum moves in both bull and bear markets.
# 1d EMA34 filter ensures alignment with daily trend to avoid counter-trend trades.
# Volume confirmation (>1.5x 20-period average) filters false breakouts.
# Designed for 12h timeframe targeting 12-37 trades/year (50-150 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for pivot calculation and trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla pivot levels for the previous day
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align pivot levels to 12h timeframe (use previous day's levels)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if data not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close above R1 + 1d uptrend + volume confirmation
            if (close[i] > r1_1d_aligned[i] and    # breakout above R1
                close[i] > ema_34_1d_aligned[i] and  # price above 1d EMA (uptrend)
                volume[i] > 1.5 * vol_avg_20[i]):   # volume spike
                signals[i] = 0.25
                position = 1
            # Short: Close below S1 + 1d downtrend + volume confirmation
            elif (close[i] < s1_1d_aligned[i] and   # breakout below S1
                  close[i] < ema_34_1d_aligned[i] and  # price below 1d EMA (downtrend)
                  volume[i] > 1.5 * vol_avg_20[i]):   # volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Camarilla level or trend reversal
            if position == 1:
                # Exit long: price returns to S1 or trend turns down
                if (close[i] < s1_1d_aligned[i] or 
                    close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price returns to R1 or trend turns up
                if (close[i] > r1_1d_aligned[i] or 
                    close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_1dEMA34_VolumeConfirm"
timeframe = "12h"
leverage = 1.0