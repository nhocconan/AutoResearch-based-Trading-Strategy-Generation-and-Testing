#!/usr/bin/env python3
# 1h_Volume_Trend_Reversal
# Hypothesis: Use 4h trend direction + 1h volume spike + price reversal from short-term extremes.
# Long when 4h uptrend + volume spike + price near 1h low; short when 4h downtrend + volume spike + price near 1h high.
# Designed for low frequency (15-35 trades/year) to avoid fee drag. Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend).

name = "1h_Volume_Trend_Reversal"
timeframe = "1h"
leverage = 1.0

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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1h RSI(14) for overbought/oversold
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 1h volume spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 2.0)  # 2x average volume
    
    # 1h price position within recent range (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    range_width = highest_20 - lowest_20
    # Avoid division by zero
    range_width_safe = np.where(range_width == 0, 1, range_width)
    position_in_range = (close - lowest_20) / range_width_safe  # 0 = low, 1 = high
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: 4h EMA50 slope
        if i >= 51:
            ema_prev = ema_50_4h_aligned[i-1]
            ema_curr = ema_50_4h_aligned[i]
            trend_up = ema_curr > ema_prev
            trend_down = ema_curr < ema_prev
        else:
            trend_up = ema_50_4h_aligned[i] > close[i]  # fallback
            trend_down = ema_50_4h_aligned[i] < close[i]
        
        # Volume filter
        vol_ok = vol_spike[i]
        
        # Price position signals
        near_low = position_in_range[i] < 0.3  # near bottom of range
        near_high = position_in_range[i] > 0.7  # near top of range
        
        # RSI filters
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        if position == 0:
            # LONG: 4h uptrend + volume spike + oversold + near low
            if trend_up and vol_ok and rsi_oversold and near_low:
                signals[i] = 0.20
                position = 1
            # SHORT: 4h downtrend + volume spike + overbought + near high
            elif trend_down and vol_ok and rsi_overbought and near_high:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # EXIT LONG: trend fails or overbought
            if not trend_up or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: trend fails or oversold
            if not trend_down or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals