#!/usr/bin/env python3
"""
12h_1W_RelativeStrength_PriceChannel
Hypothesis: On 12h timeframe, take long positions when price breaks above weekly Donchian high
with volume confirmation and weekly relative strength (vs BTC), and short when breaks below
weekly Donchian low with volume confirmation and weekly relative weakness. Uses weekly trend
filter to avoid counter-trend trades. Designed for low trade frequency (~15-25/year) to
minimize fee drag while capturing strong momentum moves in both bull and bear markets.
Weekly relative strength helps identify outperforming/underperforming assets during regime shifts.
"""

name = "12h_1W_RelativeStrength_PriceChannel"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for Donchian channels and relative strength
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly Donchian Channel (20-period) ---
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate rolling max/min with proper handling
    donchian_high = np.full(len(high_1w), np.nan)
    donchian_low = np.full(len(low_1w), np.nan)
    
    for i in range(20, len(high_1w)):
        donchian_high[i] = np.max(high_1w[i-20:i])
        donchian_low[i] = np.min(low_1w[i-20:i])
    
    # --- Weekly Relative Strength vs BTC ---
    # Calculate 12-period RSI for weekly close to measure momentum
    close_1w = df_1w['close'].values
    rsi_period = 12
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close_1w), np.nan)
    avg_loss = np.full(len(close_1w), np.nan)
    
    # Wilder's smoothing
    for i in range(len(close_1w)):
        if i < rsi_period:
            if i == 0:
                avg_gain[i] = gain[i]
                avg_loss[i] = loss[i]
            else:
                avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
                avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
        elif i == rsi_period:
            avg_gain[i] = np.mean(gain[0:rsi_period])
            avg_loss[i] = np.mean(loss[0:rsi_period])
        else:
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # --- Weekly Trend Filter (EMA34) ---
    ema_34 = np.full(len(close_1w), np.nan)
    for i in range(len(close_1w)):
        if i < 34:
            if i == 0:
                ema_34[i] = close_1w[i]
            else:
                ema_34[i] = (close_1w[i] * 2 / (34 + 1)) + (ema_34[i-1] * (33 / (34 + 1)))
        else:
            ema_34[i] = (close_1w[i] * 2 / (34 + 1)) + (ema_34[i-1] * (33 / (34 + 1)))
    
    # Align weekly indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # --- 12h Volume Confirmation (20-period average) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max(weekly Donchian needs 20, RSI needs 12, EMA34 needs 34, volume MA needs 20)
    start_idx = max(20, 12, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(rsi_1w_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > vol_ma[i] * 1.5
        
        if position == 0:
            # Long: break above weekly Donchian high with RS > 50 and volume spike
            if (close[i] > donchian_high_aligned[i] and 
                rsi_1w_aligned[i] > 50 and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: break below weekly Donchian low with RS < 50 and volume spike
            elif (close[i] < donchian_low_aligned[i] and 
                  rsi_1w_aligned[i] < 50 and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price breaks below weekly Donchian low OR weekly EMA34 turns down
                if (close[i] < donchian_low_aligned[i] or 
                    ema_34_aligned[i] < ema_34_aligned[i-1]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above weekly Donchian high OR weekly EMA34 turns up
                if (close[i] > donchian_high_aligned[i] or 
                    ema_34_aligned[i] > ema_34_aligned[i-1]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals