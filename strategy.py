#!/usr/bin/env python3
# 1H_Camarilla_R1_S1_Breakout_4hTrend_1dVolume
# Hypothesis: Camarilla pivot breakout on 1h with 4h trend filter (close > EMA50) and 1d volume spike confirmation.
# Uses tight entry conditions to avoid overtrading: only trade when price breaks above R1 or below S1,
# 4h trend confirms direction, and 1d volume is above 1.5x 20-period average.
# Designed for 15-37 trades/year on 1h timeframe with position size 0.20.
# Works in both bull and bear markets by using trend filter and volume confirmation.

name = "1H_Camarilla_R1_S1_Breakout_4hTrend_1dVolume"
timeframe = "1h"
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
    
    # Get 4h data for trend (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend direction
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume average (20-period)
    vol_avg_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    # Calculate Camarilla levels for 1h using previous day's OHLC
    # We'll use daily OHLC from 1d data to compute Camarilla levels for 1h
    # Camarilla levels: 
    # H4 = close + 1.5*(high-low)
    # H3 = close + 1.1*(high-low)
    # H2 = close + 0.6*(high-low)
    # H1 = close + 0.382*(high-low)
    # L1 = close - 0.382*(high-low)
    # L2 = close - 0.6*(high-low)
    # L3 = close - 1.1*(high-low)
    # L4 = close - 1.5*(high-low)
    # We focus on H1 (R1) and L1 (S1) for breakouts
    
    # Get previous day's OHLC from 1d data (shifted by 1 to avoid look-ahead)
    prev_day_open = df_1d['open'].shift(1).values
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    prev_day_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla R1 and S1 from previous day's data
    camarilla_r1 = prev_day_close + 0.382 * (prev_day_high - prev_day_low)
    camarilla_s1 = prev_day_close - 0.382 * (prev_day_high - prev_day_low)
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate 1h EMA(20) for entry timing filter (optional, but helps avoid whipsaws)
    ema_20_1h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_avg_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 4h EMA50
        price_above_ema = close[i] > ema_50_4h_aligned[i]
        price_below_ema = close[i] < ema_50_4h_aligned[i]
        
        # Volume filter: current 1d volume > 1.5x 20-period average
        volume_spike = volume[i] > 1.5 * vol_avg_20_aligned[i]
        
        # Camarilla breakout conditions
        breakout_above_r1 = close[i] > camarilla_r1_aligned[i]
        breakout_below_s1 = close[i] < camarilla_s1_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above R1, 4h uptrend, volume spike
            if (breakout_above_r1 and 
                price_above_ema and 
                volume_spike):
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below S1, 4h downtrend, volume spike
            elif (breakout_below_s1 and 
                  price_below_ema and 
                  volume_spike):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 or 4h trend turns down
            if (breakout_below_s1 or not price_above_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above R1 or 4h trend turns up
            if (breakout_above_r1 or not price_below_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals