# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
6h_WeeklyPivot_Bias_DailyBreakout_Volume
Hypothesis: Use weekly pivot bias (from 1w) to filter daily breakout signals on 6h timeframe.
In bull markets: weekly bias long + daily breakout long = high probability trend continuation.
In bear markets: weekly bias short + daily breakdown short = high probability trend continuation.
Weekly pivot provides structural bias, daily breakout provides entry timing, volume confirms conviction.
Designed to work in both bull and bear regimes by aligning with higher timeframe trend.
Target: 50-150 total trades over 4 years = 12-37/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1w data (Weekly bias) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot points (using prior week's OHLC)
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    # R2 = P + (H - L)
    # S2 = P - (H - L)
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pp_1w - low_1w
    s1_1w = 2 * pp_1w - high_1w
    r2_1w = pp_1w + (high_1w - low_1w)
    s2_1w = pp_1w - (high_1w - low_1w)
    
    # Weekly bias: price > R1 = bullish bias, price < S1 = bearish bias
    weekly_bullish = pp_1w > s1_1w  # Actually, we'll use price position relative to pivot
    weekly_bias_bullish = close_1w > pp_1w
    weekly_bias_bearish = close_1w < pp_1w
    
    # Align weekly bias to 6h
    weekly_bias_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias_bullish.astype(float))
    weekly_bias_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias_bearish.astype(float))
    
    # === 1d data (Daily breakout levels) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily pivot points (using prior day's OHLC)
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pp_1d - low_1d
    s1_1d = 2 * pp_1d - high_1d
    r2_1d = pp_1d + (high_1d - low_1d)
    s2_1d = pp_1d - (high_1d - low_1d)
    
    # Align daily levels to 6h
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # === 6h indicators for entry timing and confirmation ===
    # RSI(14) for momentum filter
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume ratio (current vs 24-period average)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = volume / vol_ma_24
    
    # ATR(6) for volatility filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=6, min_periods=6).mean().values
    
    # Session filter: 08-20 UTC (active trading hours)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(r2_1d_aligned[i]) or np.isnan(s2_1d_aligned[i]) or
            np.isnan(weekly_bias_bullish_aligned[i]) or np.isnan(weekly_bias_bearish_aligned[i]) or
            np.isnan(rsi[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        price = close[i]
        pp_1d = pp_1d_aligned[i]
        r1_1d = r1_1d_aligned[i]
        s1_1d = s1_1d_aligned[i]
        r2_1d = r2_1d_aligned[i]
        s2_1d = s2_1d_aligned[i]
        weekly_bullish = weekly_bias_bullish_aligned[i] > 0.5
        weekly_bearish = weekly_bias_bearish_aligned[i] > 0.5
        rsi_val = rsi[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price breaks below daily S1 OR RSI becomes overbought
            if (price < s1_1d) or (rsi_val > 70):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price breaks above daily R1 OR RSI becomes oversold
            if (price > r1_1d) or (rsi_val < 30):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Only trade during session
            if in_session:
                # LONG: Weekly bullish bias AND price breaks above daily R1 AND volume spike
                if weekly_bullish and (price > r1_1d) and (rsi_val < 60) and (vol_ratio_val > 1.8):
                    signals[i] = 0.25
                    position = 1
                    continue
                
                # SHORT: Weekly bearish bias AND price breaks below daily S1 AND volume spike
                elif weekly_bearish and (price < s1_1d) and (rsi_val > 40) and (vol_ratio_val > 1.8):
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WeeklyPivot_Bias_DailyBreakout_Volume"
timeframe = "6h"
leverage = 1.0