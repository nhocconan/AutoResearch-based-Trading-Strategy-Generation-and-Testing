#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeS_v5
# Hypothesis: Further tighten entry conditions by requiring volume spike >3x average and
# adding ADX(14) > 25 trend filter to reduce false breakouts. Maintains 1d EMA34 trend filter.
# Target: 15-25 trades/year to minimize fee drag while improving win rate in both bull/bear.

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeS_v5"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels: R1, S1
    camarilla_range = high_1d - low_1d
    r1 = close_1d + 1.1 * camarilla_range / 12
    s1 = close_1d - 1.1 * camarilla_range / 12
    
    # Get 1d data for trend filter (EMA34)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get 1d data for ADX trend filter
    # Calculate TR, +DM, -DM
    high_diff = np.diff(high_1d, prepend=high_1d[0])
    low_diff = np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(np.roll(high_1d, 1) - close_1d)
    tr3 = np.abs(np.roll(low_1d, 1) - close_1d)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * (14-1) + tr[i]) / 14
    
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / atr)
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align all indicators to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    ema_34_1d_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    adx_4h = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume spike filter on 4h (30-period average)
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (3.0 * vol_ma_30)  # Increased threshold
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    for i in range(60, n):
        # Skip if any critical value is NaN
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(ema_34_1d_4h[i]) or np.isnan(adx_4h[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: Price > R1, above 1d EMA34 trend, ADX > 25, volume spike
            if (close[i] > r1_4h[i] and close[i] > ema_34_1d_4h[i] and 
                adx_4h[i] > 25 and volume_spike[i]):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: Price < S1, below 1d EMA34 trend, ADX > 25, volume spike
            elif (close[i] < s1_4h[i] and close[i] < ema_34_1d_4h[i] and 
                  adx_4h[i] > 25 and volume_spike[i]):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position == 1:
            # Exit conditions: require minimum 4 bars held
            if bars_since_entry >= 4:
                if close[i] < r1_4h[i] or close[i] < ema_34_1d_4h[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:
                # Hold position for minimum period
                signals[i] = 0.25
        elif position == -1:
            # Exit conditions: require minimum 4 bars held
            if bars_since_entry >= 4:
                if close[i] > s1_4h[i] or close[i] > ema_34_1d_4h[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
            else:
                # Hold position for minimum period
                signals[i] = -0.25
    
    return signals