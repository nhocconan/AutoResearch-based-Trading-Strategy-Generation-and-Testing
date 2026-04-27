#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray with 12h trend filter and volume confirmation
# Elder Ray measures bull/bear power using EMA(13) and price extremes.
# Bull Power = High - EMA13, Bear Power = Low - EMA13
# In trending markets, buy when Bull Power turns positive with divergence,
# sell when Bear Power turns negative with divergence.
# Trend filter from 12h EMA(34) ensures we trade with higher timeframe trend.
# Volume filter confirms conviction. Works in bull/bear via 12h trend filter.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Elder Ray: EMA13 of close, Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Align Elder Ray to 6h timeframe (same timeframe, just ensure proper alignment)
    bull_power_aligned = align_htf_to_ltf(prices, prices, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, prices, bear_power)
    
    # 12h EMA trend filter (34-period)
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume filter: volume > 1.5 x 20-period average (~5 days of 6h bars)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need EMA13 (13), EMA34 (34), volume MA (20)
    start_idx = max(13, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: significant volume spike
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend filter from 12h EMA
        bullish_trend = price > ema_34_aligned[i]
        bearish_trend = price < ema_34_aligned[i]
        
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        
        # Divergence conditions: power changing sign with trend
        bull_divergence = bull_power_val > 0 and bear_power_val < 0  # Bullish divergence
        bear_divergence = bear_power_val < 0 and bull_power_val > 0  # Bearish divergence (same condition)
        # Actually we need: bull power turning positive, bear power turning negative
        # For simplicity, use zero-cross with hysteresis
        if i > start_idx:
            bull_power_prev = bull_power_aligned[i-1]
            bear_power_prev = bear_power_aligned[i-1]
            bull_cross_up = bull_power_prev <= 0 and bull_power_val > 0
            bear_cross_down = bear_power_prev >= 0 and bear_power_val < 0
        else:
            bull_cross_up = False
            bear_cross_down = False
        
        if position == 0:
            # Long: bull power crossing up + volume + bullish 12h trend
            if bull_cross_up and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: bear power crossing down + volume + bearish 12h trend
            elif bear_cross_down and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: bull power turns negative or trend turns bearish
            if bull_power_val <= 0 or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: bear power turns positive or trend turns bullish
            if bear_power_val >= 0 or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ElderRay_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0