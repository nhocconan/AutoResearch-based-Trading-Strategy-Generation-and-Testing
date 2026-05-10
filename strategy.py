#!/usr/bin/env python3
# 1h_4h_1d_Camarilla_R1_S1_Breakout_VolumeTrend
# Hypothesis: Combine Camarilla pivot levels (R1/S1) with 4h trend (EMA34) and 1d volume confirmation for 1h entries.
# Uses higher timeframes for signal direction (4h trend, 1d volume) and 1h for precise entry timing.
# Camarilla levels provide statistically significant support/resistance; volume confirms institutional participation.
# Designed for low trade frequency (15-35/year) to minimize fee drag in both bull and bear markets.

name = "1h_4h_1d_Camarilla_R1_S1_Breakout_VolumeTrend"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 35:
        return np.zeros(n)
    
    # Get 1d data for Camarilla levels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h EMA34 for trend filter
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_slope_34_4h = np.diff(ema_34_4h, prepend=ema_34_4h[0])  # today - yesterday
    ema_slope_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_slope_34_4h)
    
    # 1d Camarilla levels (R1, S1)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_width = 1.1 * (high_1d - low_1d) / 12
    r1_1d = close_1d + camarilla_width
    s1_1d = close_1d - camarilla_width
    
    # Align Camarilla levels to 1h timeframe (use previous day's levels)
    r1_1d_prev = np.roll(r1_1d, 1)
    s1_1d_prev = np.roll(s1_1d, 1)
    r1_1d_prev[0] = r1_1d[0]
    s1_1d_prev[0] = s1_1d[0]
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d_prev)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d_prev)
    
    # 1d volume confirmation (1.5x 20-day average)
    vol_ma_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # ATR for risk management
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_slope_34_4h_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from 4h EMA34 slope
        bullish_trend = ema_slope_34_4h_aligned[i] > 0
        bearish_trend = ema_slope_34_4h_aligned[i] < 0
        
        # Volume confirmation (1.5x 20-day average)
        volume_confirm = volume[i] > 1.5 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Long: break above R1 in bullish trend with volume confirmation
            if close[i] > r1_aligned[i] and bullish_trend and volume_confirm:
                signals[i] = 0.20
                position = 1
            # Short: break below S1 in bearish trend with volume confirmation
            elif close[i] < s1_aligned[i] and bearish_trend and volume_confirm:
                signals[i] = -0.20
                position = -1
        else:
            if position == 1:
                # Exit: close below S1 or trend turns bearish
                if close[i] < s1_aligned[i] or not bullish_trend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit: close above R1 or trend turns bullish
                if close[i] > r1_aligned[i] or not bearish_trend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals