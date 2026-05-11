#!/usr/bin/env python3
"""
1h_4d_PriceAction_Trend
Hypothesis: Price action with 4h trend filter and daily volume confirmation.
- Long when: Price > 4h EMA50, daily volume > 20-day average, and bullish engulfing candle
- Short when: Price < 4h EMA50, daily volume > 20-day average, and bearish engulfing candle
- Exit when price crosses 4h EMA50 or volume condition fails
Uses 4h for trend direction, 1h for precise entry timing, and daily volume for conviction.
Targets 15-30 trades/year (60-120 over 4 years) to minimize fee drag.
"""

name = "1h_4d_PriceAction_Trend"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1h OHLCV
    close_1h = prices['close'].values
    open_1h = prices['open'].values
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    volume_1h = prices['volume'].values
    
    # --- 4h Trend Filter: EMA50 ---
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # --- Daily Volume Filter: 20-day average ---
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # --- Bullish/Bearish Engulfing Detection ---
    bullish_engulf = np.zeros(n, dtype=bool)
    bearish_engulf = np.zeros(n, dtype=bool)
    for i in range(1, n):
        # Bullish engulf: current green candle engulfs previous red candle
        if close_1h[i] > open_1h[i] and close_1h[i-1] < open_1h[i-1]:
            if close_1h[i] >= open_1h[i-1] and open_1h[i] <= close_1h[i-1]:
                bullish_engulf[i] = True
        # Bearish engulf: current red candle engulfs previous green candle
        if close_1h[i] < open_1h[i] and close_1h[i-1] > open_1h[i-1]:
            if open_1h[i] >= close_1h[i-1] and close_1h[i] <= open_1h[i-1]:
                bearish_engulf[i] = True
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50  # for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 4h trend
        trend_up = close_1h[i] > ema50_4h_aligned[i]
        trend_down = close_1h[i] < ema50_4h_aligned[i]
        
        # Volume confirmation: current 1h volume > daily 20-day average volume
        # Scale daily volume to hourly approximation (divide by 24 for 24h in a day)
        vol_ok = volume_1h[i] > (vol_ma_20_1d_aligned[i] / 24.0)
        
        if position == 0:
            # Look for entries only in direction of 4h trend with volume and price action
            if bullish_engulf[i] and trend_up and vol_ok:
                # Long: bullish engulfing + 4h uptrend + volume
                signals[i] = 0.20
                position = 1
            elif bearish_engulf[i] and trend_down and vol_ok:
                # Short: bearish engulfing + 4h downtrend + volume
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price crosses below 4h EMA50 OR volume fails
                if not trend_up or not vol_ok:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit short: price crosses above 4h EMA50 OR volume fails
                if not trend_down or not vol_ok:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals