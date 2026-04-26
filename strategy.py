#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_ATRVolFilter_v1
Hypothesis: Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume-ATR confirmation (>1.5x volume MA AND ATR>0.5*ATR(20)). 
Long when price breaks above R3 in 1d uptrend with confirmation. Short when price breaks below S3 in 1d downtrend with confirmation.
Uses discrete position sizing (0.25) to minimize fee churn. Camarilla levels derived from prior 1d OHLC.
Designed to work in both bull and bear markets by following the 1d trend. Tight entry conditions target ~25-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 1d OHLC
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Shift by 1 to use prior day's OHLC for current day's levels
    close_1d_prev = np.roll(close_1d, 1)
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev[0] = np.nan
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    
    # Camarilla R3, S3 levels (wider breakout bands)
    camarilla_range = high_1d_prev - low_1d_prev
    r3 = close_1d_prev + camarilla_range * 1.1 / 4
    s3 = close_1d_prev - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    uptrend_1d = close > ema_34_1d_aligned
    downtrend_1d = close < ema_34_1d_aligned
    
    # Volume confirmation: volume > 1.5x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # ATR filter: ATR > 0.5 * ATR(20) to avoid low-volatility chop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    atr_filter = atr > (0.5 * atr_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for 1d EMA + 20 for volume MA + 20 for ATR + 1 for Camarilla shift)
    start_idx = 55
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(atr_filter[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price breaks above R3 with 1d uptrend and volume/ATR confirmation
            if (close[i] > r3_aligned[i] and 
                uptrend_1d[i] and volume_spike[i] and atr_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with 1d downtrend and volume/ATR confirmation
            elif (close[i] < s3_aligned[i] and 
                  downtrend_1d[i] and volume_spike[i] and atr_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below S3 (strong reversal) OR 1d trend changes to downtrend
            if (close[i] < s3_aligned[i] or not uptrend_1d[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above R3 (strong reversal) OR 1d trend changes to uptrend
            if (close[i] > r3_aligned[i] or not downtrend_1d[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_ATRVolFilter_v1"
timeframe = "4h"
leverage = 1.0