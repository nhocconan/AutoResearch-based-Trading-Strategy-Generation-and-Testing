#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly Bollinger Band breakout with daily RSI confirmation and volume filter.
# Buy when price breaks above upper BB(20,2) on weekly timeframe and daily RSI < 70 (avoid overbought).
# Sell when price breaks below lower BB(20,2) on weekly timeframe and daily RSI > 30 (avoid oversold).
# Weekly timeframe reduces whipsaw, daily RSI adds momentum filter, volume confirms breakout strength.
# Works in trending markets by catching breakouts; avoids false signals in ranging markets via RSI filter.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly data for Bollinger Bands ===
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_volume = df_1w['volume'].values
    
    # Weekly Bollinger Bands (20, 2)
    weekly_close_series = pd.Series(weekly_close)
    bb_middle = weekly_close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = weekly_close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Align to daily timeframe
    bb_middle_aligned = align_htf_to_ltf(prices, df_1w, bb_middle)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1w, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1w, bb_lower)
    
    # Daily RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    avg_gain = gain_series.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = loss_series.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily volume average (20-period)
    volume_series = pd.Series(volume)
    vol_avg20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if np.isnan(bb_middle_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_avg20[i]):
            continue
            
        # Volume filter: current volume > 1.5x 20-day average
        vol_filter = volume[i] > 1.5 * vol_avg20[i]
        
        # Long entry: price above weekly upper BB + RSI not overbought + volume
        if close[i] > bb_upper_aligned[i] and rsi[i] < 70 and vol_filter:
            if position <= 0:
                signals[i] = 0.25
                position = 1
        # Short entry: price below weekly lower BB + RSI not oversold + volume
        elif close[i] < bb_lower_aligned[i] and rsi[i] > 30 and vol_filter:
            if position >= 0:
                signals[i] = -0.25
                position = -1
        # Exit conditions: opposite BB touch or RSI extreme
        elif position == 1:
            if close[i] < bb_middle_aligned[i] or rsi[i] > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if close[i] > bb_middle_aligned[i] or rsi[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyBB_RSI_VolumeFilter"
timeframe = "1d"
leverage = 1.0