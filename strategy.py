#!/usr/bin/env python3
# 4h_HTF_Breakout_Trend_Volume_1d
# Hypothesis: Use 4h Donchian breakout for entry, filtered by 1d EMA50 trend and volume spike.
# Long: price breaks above Donchian(20) upper band + price > 1d EMA50 + volume > 1.5x avg.
# Short: price breaks below Donchian(20) lower band + price < 1d EMA50 + volume > 1.5x avg.
# Exit: price returns to Donchian midline (average of upper/lower).
# Designed for low frequency (15-30 trades/year) to avoid fee drag. Works in bull (catch breakouts)
# and bear (catch breakdowns) with trend filter and volume confirmation.

name = "4h_HTF_Breakout_Trend_Volume_1d"
timeframe = "4h"
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
    
    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period) on 4h data
    period = 20
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    
    for i in range(n):
        if i < period:
            upper_band[i] = np.max(high[max(0, i-period+1):i+1])
            lower_band[i] = np.min(low[max(0, i-period+1):i+1])
        else:
            upper_band[i] = np.max(high[i-period+1:i+1])
            lower_band[i] = np.min(low[i-period+1:i+1])
    
    midline = (upper_band + lower_band) / 2
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align daily data to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(midline[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA50
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i] * 1.5  # 50% above average
        
        # Breakout conditions
        buy_breakout = close[i] > upper_band[i]
        sell_breakout = close[i] < lower_band[i]
        
        # Exit conditions: return to midline
        exit_long = close[i] < midline[i]
        exit_short = close[i] > midline[i]
        
        if position == 0:
            # LONG: breakout above upper band + uptrend + volume spike
            if buy_breakout and trend_up and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: breakdown below lower band + downtrend + volume spike
            elif sell_breakout and trend_down and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: price returns to midline
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price returns to midline
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals