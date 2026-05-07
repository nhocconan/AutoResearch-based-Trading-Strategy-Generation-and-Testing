#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_1DTrend_VolumeBreakout_v1
# Hypothesis: 12-hour strategy using daily Camarilla R1/S1 breakouts with 1-day EMA34 trend filter and volume spike confirmation.
# Uses tighter profit targets and stricter volume confirmation to reduce trade frequency and improve edge.
# Designed to work in both bull and bear markets by using trend filter to avoid counter-trend trades.
# Targets 15-35 trades/year to minimize fee drag. Uses discrete position sizing (0.25).

name = "12h_Camarilla_R1_S1_1DTrend_VolumeBreakout_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough data for EMA34
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels for each day
    range_1d = prev_high - prev_low
    r1 = prev_close + range_1d * 1.1 / 12
    s1 = prev_close - range_1d * 1.1 / 12
    pp = (prev_high + prev_low + prev_close) / 3  # Pivot point
    
    # Calculate 1-day EMA34 for trend filter
    ema_34 = pd.Series(prev_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align Camarilla levels, EMA, and pivot to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume filter: current volume > 2.0x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Volatility filter: avoid low volatility periods (ATR < 0.3% of price)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    vol_filter = atr > 0.003 * close  # ATR > 0.3% of price
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure we have EMA34 and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(ema_34_aligned[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or np.isnan(vol_filter[i]) or not vol_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation (2.0x average volume)
        volume_filter = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above R1 + uptrend (price > EMA34) + volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_34_aligned[i] and   # Uptrend filter
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 + downtrend (price < EMA34) + volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_34_aligned[i] and   # Downtrend filter
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit conditions:
            # 1. Price returns to pivot point (mean reversion)
            # 2. Opposite S1/R1 level break (tighter exit for profit protection)
            at_pivot = abs(close[i] - pp_aligned[i]) < (r1_aligned[i] - pp_aligned[i]) * 0.10  # Within 10% of PP
            opposite_break = (position == 1 and close[i] < s1_aligned[i]) or \
                           (position == -1 and close[i] > r1_aligned[i])
            
            if at_pivot or opposite_break:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals