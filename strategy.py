#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_Volume
Hypothesis: Camarilla pivot level breaks (R1/S1) on 4h provide high-probability entries when aligned with 12h EMA50 trend and volume spikes. This strategy captures institutional breakout attempts with defined risk via ATR stops. Designed for 20-40 trades/year per symbol to minimize fee drag while capturing strong directional moves in both bull and bear markets.
"""

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- 12h Trend Filter: EMA50 ---
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # --- Daily Data for Camarilla Pivots (using previous day's OHLC) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    prev_day_close = df_1d['close'].values
    prev_day_high = df_1d['high'].values
    prev_day_low = df_1d['low'].values
    
    camarilla_width = (prev_day_high - prev_day_low) * 1.1 / 12
    r1_level = prev_day_close + camarilla_width
    s1_level = prev_day_close - camarilla_width
    
    # Align Camarilla levels to 4h timeframe (using previous day's close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_level)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_level)
    
    # --- Volume Filter: spike above 1.5x median of last 20 periods ---
    vol_median = pd.Series(volume_4h).rolling(window=20, min_periods=10).median().values
    vol_threshold = vol_median * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 20  # for volume median and EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_threshold[i])):
            if position != 0:
                # Check stoploss
                if position == 1 and close_4h[i] <= entry_price - 2.0 * (high_4h[i] - low_4h[i]):
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_4h[i] >= entry_price + 2.0 * (high_4h[i] - low_4h[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine 12h trend
        trend_up = close_4h[i] > ema50_12h_aligned[i]
        trend_down = close_4h[i] < ema50_12h_aligned[i]
        
        # Volume filter: spike above 1.5x median
        vol_ok = volume_4h[i] > vol_threshold[i]
        
        if position == 0:
            # Look for entries only in direction of 12h trend with volume spike
            if close_4h[i] > r1_aligned[i] and trend_up and vol_ok:
                # Long: price breaks above R1 + 12h uptrend + volume spike
                signals[i] = 0.25
                position = 1
                entry_price = close_4h[i]
            elif close_4h[i] < s1_aligned[i] and trend_down and vol_ok:
                # Short: price breaks below S1 + 12h downtrend + volume spike
                signals[i] = -0.25
                position = -1
                entry_price = close_4h[i]
        else:
            # Update stoploss and check exits
            if position == 1:
                # Stoploss: 2x ATR equivalent
                atr_approx = high_4h[i] - low_4h[i]
                if close_4h[i] <= entry_price - 2.0 * atr_approx:
                    signals[i] = 0.0
                    position = 0
                # Exit: price returns to or below previous day's close (pivot point)
                elif close_4h[i] <= df_1d['close'].values[min(i//6, len(df_1d)-1)] if i//6 < len(df_1d) else df_1d['close'].values[-1]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Stoploss: 2x ATR equivalent
                atr_approx = high_4h[i] - low_4h[i]
                if close_4h[i] >= entry_price + 2.0 * atr_approx:
                    signals[i] = 0.0
                    position = 0
                # Exit: price returns to or above previous day's close (pivot point)
                elif close_4h[i] >= df_1d['close'].values[min(i//6, len(df_1d)-1)] if i//6 < len(df_1d) else df_1d['close'].values[-1]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals