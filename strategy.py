# 12h_Camarilla_Pivot_S1_R1_Breakout_1wEMA34_Trend_VolumeConfirm_v1
# Strategy: Daily Camarilla Pivot Points (S1/R1) breakout with weekly EMA34 trend filter and volume confirmation
# Target: 12h timeframe with strict entry conditions to limit trades to 12-37/year
# Logic: Long when price breaks above R1 with volume and weekly uptrend; Short when breaks below S1 with volume and weekly downtrend
# Exit when price returns to opposite pivot level or trend reverses
# Weekly EMA34 ensures alignment with higher timeframe trend to avoid counter-trend trades
# Volume confirmation filters out low-conviction breakouts
# Works in both bull (trend continuation) and bear (mean reversion at extreme levels) markets

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for Camarilla pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (based on previous day)
    # Pivot = (H + L + C) / 3
    # S1 = C - (H - L) * 1.1 / 12
    # R1 = C + (H - L) * 1.1 / 12
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    s1_1d = close_1d - (range_1d * 1.1 / 12.0)
    r1_1d = close_1d + (range_1d * 1.1 / 12.0)
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    
    # Load weekly data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume spike filter (24-period on 12h = roughly 12 days)
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > 1.5 * vol_ma24  # Require 1.5x volume for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R1 with volume and weekly uptrend
            if close[i] > r1_aligned[i] and vol_spike[i] and close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with volume and weekly downtrend
            elif close[i] < s1_aligned[i] and vol_spike[i] and close[i] < ema34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite level or trend reversal vs weekly EMA34
            if position == 1:
                if close[i] < s1_aligned[i] or close[i] < ema34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > r1_aligned[i] or close[i] > ema34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_Pivot_S1_R1_Breakout_1wEMA34_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0