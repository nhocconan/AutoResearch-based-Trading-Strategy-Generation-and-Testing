#!/usr/bin/env python3
"""
4h Camarilla R1/S1 Breakout + 12h EMA50 Trend + Volume Spike
Hypothesis: Camarilla pivot levels (R1/S1) from prior day act as intraday support/resistance.
A break above R1 with volume and 12h uptrend signals bullish momentum; break below S1 with
volume and 12h downtrend signals bearish momentum. Uses 4h timeframe for balance of trade
frequency and signal quality. Works in bull/bear markets via 12h EMA trend filter.
Volume confirmation reduces false breakouts. Discrete sizing (0.25) minimizes fee churn.
"""

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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for Camarilla pivot calculation (prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels using prior 1d bar's OHLC
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C = close, H = high, L = low of prior day
    prior_close = df_1d['close'].shift(1).values  # prior day close
    prior_high = df_1d['high'].shift(1).values    # prior day high
    prior_low = df_1d['low'].shift(1).values      # prior day low
    
    camarilla_range = (prior_high - prior_low) * 1.1 / 12.0
    r1 = prior_close + camarilla_range
    s1 = prior_close - camarilla_range
    
    # Align Camarilla levels to 4h timeframe (already completed prior day)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA50 warmup and prior day data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_50_aligned[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        
        # Volume spike: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 1.5 * vol_ma_20
        
        if position == 0:
            # Long: price breaks above R1 AND above 12h EMA50 (uptrend filter)
            long_condition = (curr_close > r1_level) and (curr_close > ema_trend) and volume_spike
            # Short: price breaks below S1 AND below 12h EMA50 (downtrend filter)
            short_condition = (curr_close < s1_level) and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: price returns below S1 or trend breaks
            if curr_close <= s1_level or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above R1 or trend breaks
            if curr_close >= r1_level or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0