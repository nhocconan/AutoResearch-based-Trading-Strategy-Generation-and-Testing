#!/usr/bin/env python3
# 12H_Camarilla_R3S3_WeeklyTrend_1DayVolume_Spike
# Hypothesis: 12-hour chart using weekly trend filter (price above/below weekly EMA34)
# combined with daily Camarilla R3/S3 breakout and volume spike confirmation.
# Weekly trend ensures we trade with the dominant multi-week direction,
# reducing counter-trend trades in choppy markets.
# Volume spike on 12h chart confirms momentum at breakout.
# Target: 15-25 trades/year to minimize fee drag while maintaining edge.
# Uses discrete position sizing (0.25) and strict entry conditions.

name = "12H_Camarilla_R3S3_WeeklyTrend_1DayVolume_Spike"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need enough data for EMA34
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least previous day
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    r3 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 2
    s3 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 2
    
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume filter on 12h chart: current volume > 2.0x average volume (30-period)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    # Volatility filter: avoid extremely low volatility (ATR < 0.2% of price)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    vol_filter = atr > 0.002 * close  # ATR > 0.2% of price
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 30)  # Ensure we have volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(vol_filter[i]) or not vol_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation (2.0x average volume)
        volume_filter = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above R3 + weekly uptrend + volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_34_1w_aligned[i] and   # Weekly uptrend filter
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 + weekly downtrend + volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_34_1w_aligned[i] and   # Weekly downtrend filter
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: Price returns to the middle of the Camarilla range (H4/L4)
            h4 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 6
            l4 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 6
            h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
            l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
            
            camarilla_mid = (h4_aligned[i] + l4_aligned[i]) / 2
            at_mid = abs(close[i] - camarilla_mid) < (h4_aligned[i] - l4_aligned[i]) * 0.25  # Within 25% of range
            
            if at_mid:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals