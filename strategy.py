#!/usr/bin/env python3
"""
4h Camarilla H3/L3 Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Camarilla H3/L3 levels act as key intraday support/resistance where institutional order flow clusters.
Breaking above H3 with volume and daily uptrend signals bullish momentum; breaking below L3 with volume and daily downtrend signals bearish momentum.
The 4h timeframe targets 20-50 trades/year to minimize fee drag while capturing multi-day swings. Daily EMA34 filter ensures alignment with higher timeframe trend.
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA34 warmup
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if EMA data not ready
        if np.isnan(ema_34_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels from previous day's OHLC
        # Camarilla uses previous day's range
        prev_day_idx = i - 1  # previous bar in 4h timeframe
        if prev_day_idx < 0:
            continue
            
        # For 4h timeframe, we need to aggregate to daily levels
        # Since we're using 4h bars, we approximate using lookback of 6 bars (24h/4h)
        lookback = 6
        if i < lookback:
            continue
            
        # Get daily high/low/close from 4h data (approximate)
        daily_high = np.max(high[i-lookback:i])
        daily_low = np.min(low[i-lookback:i])
        daily_close = close[i-1]  # previous close
        
        # Calculate Camarilla levels
        daily_range = daily_high - daily_low
        if daily_range <= 0:
            continue
            
        H3 = daily_close + daily_range * 1.1 / 4
        L3 = daily_close - daily_range * 1.1 / 4
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_34_aligned[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Breakout signals with trend filter
        if position == 0:
            # Long: price breaks above H3 AND above daily EMA34 (uptrend filter)
            long_condition = (curr_close > H3) and (curr_close > ema_trend) and volume_spike
            # Short: price breaks below L3 AND below daily EMA34 (downtrend filter)
            short_condition = (curr_close < L3) and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below L3 or trend breaks
            if curr_close < L3 or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above H3 or trend breaks
            if curr_close > H3 or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0