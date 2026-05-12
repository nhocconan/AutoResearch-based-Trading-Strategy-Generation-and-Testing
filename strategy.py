#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Use Camarilla pivot levels (R3/S3) from 1d to trade breakouts, filtered by 1d EMA34 trend and volume spike.
# Designed for low frequency (12-37 trades/year) to avoid fee drag. Camarilla levels provide strong support/resistance
# in ranging markets, while breakouts capture trends. The 1d EMA34 filter ensures alignment with daily trend,
# and volume confirmation reduces false breakouts. Works in both bull (breakouts) and bear (mean reversion at levels) markets.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """
    Calculate Camarilla pivot levels for a given period.
    Returns R3, R2, R1, PP, S1, S2, S3.
    """
    typical = (high + low + close) / 3
    range_ = high - low
    R3 = close + range_ * 1.1 / 4
    R2 = close + range_ * 1.1 / 2
    R1 = close + range_ * 1.1 / 6
    PP = typical
    S1 = close - range_ * 1.1 / 6
    S2 = close - range_ * 1.1 / 2
    S3 = close - range_ * 1.1 / 4
    return R3, R2, R1, PP, S1, S2, S3

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla, EMA, and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily Camarilla levels (R3/S3)
    R3_1d, _, _, _, _, _, S3_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Calculate daily EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate daily average volume (20-period) for volume spike filter
    volume_1d_series = pd.Series(volume_1d)
    avg_vol_20_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    
    # Align daily indicators to 12h timeframe
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    avg_vol_20_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA34 to be stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(R3_1d_aligned[i]) or np.isnan(S3_1d_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(avg_vol_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume spike: current 12h volume > 1.5 * daily average volume
        vol_spike = volume[i] > 1.5 * avg_vol_20_1d_aligned[i]
        
        if position == 0:
            # LONG: Close above R3 AND EMA34 uptrend AND volume spike
            if close[i] > R3_1d_aligned[i] and close[i] > ema_34_1d_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: Close below S3 AND EMA34 downtrend AND volume spike
            elif close[i] < S3_1d_aligned[i] and close[i] < ema_34_1d_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Close below EMA34 OR volume spike fades
            if close[i] < ema_34_1d_aligned[i] or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above EMA34 OR volume spike fades
            if close[i] > ema_34_1d_aligned[i] or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals