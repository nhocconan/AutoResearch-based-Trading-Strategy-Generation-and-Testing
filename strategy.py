#!/usr/bin/env python3
# 4h_1d_1w_Camarilla_R1_S1_Breakout_Volume_Spike
# Hypothesis: Combines weekly trend filter with daily Camarilla R1/S1 breakouts and volume confirmation.
# Weekly trend filters ensure we only trade in strong directional markets.
# Daily Camarilla R1/S1 levels provide tight entry points with minimal slippage.
# Volume spike confirms institutional participation, reducing false breakouts.
# Target: 20-30 trades/year to minimize fee drag while capturing meaningful moves.
# Works in both bull and bear markets by aligning with higher timeframe trend.

name = "4h_1d_1w_Camarilla_R1_S1_Breakout_Volume_Spike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1-day and 1-week data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 2 or len(df_1w) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly EMA50 for trend filter ---
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # --- Daily ATR for volatility filter ---
    atr_period = 14
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1d = tr.ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # --- Daily Camarilla levels (R1, S1) from previous day ---
    prev_1d_high = df_1d['high'].values
    prev_1d_low = df_1d['low'].values
    prev_1d_close = df_1d['close'].values
    
    camarilla_width = (prev_1d_high - prev_1d_low) * 1.1 / 12.0  # R1/S1 level
    camarilla_r1 = prev_1d_close + camarilla_width
    camarilla_s1 = prev_1d_close - camarilla_width
    
    # Align daily Camarilla levels to 4h
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # --- Volume confirmation (2x 20-period average on 4h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for weekly EMA50 (50 weeks) and 20-period volume MA
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        # Volatility filter: only trade when volatility is above average
        vol_filter = atr_14_1d_aligned[i] > np.nanmedian(atr_14_1d_aligned[:i+1])
        
        if position == 0:
            # Long: price breaks above R1 with volume surge, weekly uptrend, and sufficient volatility
            if (close[i] > camarilla_r1_aligned[i] and 
                volume_surge and 
                ema_50_1w_aligned[i] < close[i] and
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume surge, weekly downtrend, and sufficient volatility
            elif (close[i] < camarilla_s1_aligned[i] and 
                  volume_surge and 
                  ema_50_1w_aligned[i] > close[i] and
                  vol_filter):
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price drops below S1 OR weekly EMA50 turns down
                if (close[i] < camarilla_s1_aligned[i] or 
                    close[i] < ema_50_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises above R1 OR weekly EMA50 turns up
                if (close[i] > camarilla_r1_aligned[i] or 
                    close[i] > ema_50_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals