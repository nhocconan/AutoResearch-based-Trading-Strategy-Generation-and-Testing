#!/usr/bin/env python3
# 1d_weekly_ema_trend_volume_v2
# Hypothesis: Follow the weekly trend using EMA crossover on 1w, with volume confirmation on 1d.
# Enter long when weekly EMA21 > EMA50 and price pulls back to EMA21 with volume spike.
# Enter short when weekly EMA21 < EMA50 and price pulls back to EMA21 with volume spike.
# Works in bull/bear by aligning with higher timeframe trend. Uses volume spike to confirm institutional participation.
# Target: 30-100 total trades over 4 years (7-25/year).

name = "1d_weekly_ema_trend_volume_v2"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly trend: EMA crossover
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA21 and EMA50 on weekly close
    close_1w = df_1w['close'].values
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Weekly trend direction: 1 if EMA21 > EMA50, -1 if EMA21 < EMA50
    trend_1w = np.where(ema21_1w > ema50_1w, 1, -1)
    
    # Align weekly indicators to daily
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Volume spike detection on 1d
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_spike = vol_ratio > 1.5  # 50% above average volume
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(trend_1w_aligned[i]) or np.isnan(ema21_1w_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Only trade in session and with volume spike
        if not (in_session[i] and vol_spike[i]):
            if position != 0:
                # Hold position until exit signal
                pass
            else:
                signals[i] = 0.0
                continue
        
        if position == 1:  # Long position
            # Exit: weekly trend turns bearish
            if trend_1w_aligned[i] == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: weekly trend turns bullish
            if trend_1w_aligned[i] == 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need strong trend (weekly EMA21 > EMA50 for long, < for short)
            if trend_1w_aligned[i] == 1:
                # Long: weekly bullish trend + price pulls back to EMA21
                if close[i] <= ema21_1w_aligned[i] * 1.02 and close[i] >= ema21_1w_aligned[i] * 0.98:
                    # Within 2% of EMA21 (pullback zone)
                    position = 1
                    signals[i] = 0.25
            elif trend_1w_aligned[i] == -1:
                # Short: weekly bearish trend + price pulls back to EMA21
                if close[i] <= ema21_1w_aligned[i] * 1.02 and close[i] >= ema21_1w_aligned[i] * 0.98:
                    # Within 2% of EMA21 (pullback zone)
                    position = -1
                    signals[i] = -0.25
    
    return signals