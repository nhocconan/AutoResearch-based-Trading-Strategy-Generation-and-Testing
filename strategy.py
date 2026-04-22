#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot S1/R1 breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla pivots identify key intraday support/resistance levels where reversals/breakouts occur.
# 1d EMA34 filter ensures we trade in direction of daily trend, reducing counter-trend trades.
# Volume confirmation (>1.5x 20-period average) filters false breakouts.
# Designed for 4h timeframe targeting 20-40 trades/year, works in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Camarilla pivots and trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous day
    # Typical price = (H + L + C)/3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Camarilla levels: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # We only need S1 and R1 for entries
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate S1 and R1 for each day
    # R1 = C + ((H-L) * 1.1/6)
    # S1 = C - ((H-L) * 1.1/6)
    camarilla_range = (high_1d - low_1d) * 1.1 / 6
    r1_levels = close_1d + camarilla_range
    s1_levels = close_1d - camarilla_range
    
    # Align S1/R1 levels to 4h timeframe (previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_levels)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_levels)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R1 + 1d uptrend + volume confirmation
            if (close[i] > r1_aligned[i] and      # break above R1 resistance
                close[i] > ema_34_1d_aligned[i] and  # price above 1d EMA (uptrend)
                volume[i] > 1.5 * vol_avg_20[i]):   # volume spike
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 + 1d downtrend + volume confirmation
            elif (close[i] < s1_aligned[i] and   # break below S1 support
                  close[i] < ema_34_1d_aligned[i] and  # price below 1d EMA (downtrend)
                  volume[i] > 1.5 * vol_avg_20[i]):   # volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Camarilla level or trend reversal
            if position == 1:
                # Exit long: price returns to S1 or trend turns down
                if (close[i] < s1_aligned[i] or 
                    close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price returns to R1 or trend turns up
                if (close[i] > r1_aligned[i] or 
                    close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_S1R1_1dEMA34_VolumeConfirm"
timeframe = "4h"
leverage = 1.0