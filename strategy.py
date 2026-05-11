#!/usr/bin/env python3
"""
1d_1w_Camarilla_R1_S1_Breakout_1wTrend_Volume
Hypothesis: Uses weekly market structure to define trend direction and daily Camarilla R1/S1 levels for entries on breakouts with volume confirmation. In bull markets (price above weekly EMA200), buy breaks above R1; in bear markets (price below weekly EMA200), sell breaks below S1. Exits on opposite level touch or trend reversal. Designed for low trade frequency (<25/year) to minimize fee drag while capturing major moves in both bull and bear regimes.
"""

name = "1d_1w_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    pivot = (high + low + close) / 3
    range_ = high - low
    r1 = close + (range_ * 1.1 / 12)
    s1 = close - (range_ * 1.1 / 12)
    return r1, s1, pivot

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Daily OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly Trend Filter (EMA200) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(
        span=200, adjust=False, min_periods=200
    ).mean().values
    ema_200_daily = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # --- Daily Camarilla Levels ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    r1_1d, s1_1d, pivot_1d = calculate_camarilla(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values
    )
    
    r1_daily = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_daily = align_htf_to_ltf(prices, df_1d, s1_1d)
    pivot_daily = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # --- Volume Spike Detection (20-day average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_daily[i]) or np.isnan(s1_daily[i]) or 
            np.isnan(pivot_daily[i]) or np.isnan(ema_200_daily[i]) or
            np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine market regime based on weekly trend
        bull_market = close[i] > ema_200_daily[i]
        bear_market = close[i] < ema_200_daily[i]
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 2.0
        
        if position == 0:
            # Long: price breaks above R1 in bull market + volume
            if bull_market and (close[i] > r1_daily[i]) and \
               (close[i-1] <= r1_daily[i-1]) and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 in bear market + volume
            elif bear_market and (close[i] < s1_daily[i]) and \
                 (close[i-1] >= s1_daily[i-1]) and volume_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price touches S1 OR trend turns bearish
                if (close[i] <= s1_daily[i]) or (close[i] < ema_200_daily[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price touches R1 OR trend turns bullish
                if (close[i] >= r1_daily[i]) or (close[i] > ema_200_daily[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals