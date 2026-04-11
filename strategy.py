#!/usr/bin/env python3
"""
12h_1d_ema_bounce_volume_v1
Strategy: 12h EMA bounce with volume confirmation and 1d trend filter
Timeframe: 12h
Leverage: 1.0
Hypothesis: Price tends to bounce off the 1d EMA50 on 12h timeframe during pullbacks in trending markets. Uses 1d EMA50 as dynamic support/resistance, enters on 12h close near EMA50 with volume confirmation (>1.5x average volume). Trend filter requires 12h price above/below 1d EMA200 to ensure alignment with higher timeframe trend. Designed to work in both bull and bear markets by trading pullbacks in the direction of the 1d trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_ema_bounce_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for dynamic support/resistance
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d EMA200 for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        ema_50 = ema_50_1d_aligned[i]
        ema_200 = ema_200_1d_aligned[i]
        
        # Trend filter: price above EMA200 for long, below for short
        uptrend = price_close > ema_200
        downtrend = price_close < ema_200
        
        # Proximity to EMA50: within 1% of EMA50
        near_ema50 = abs(price_close - ema_50) / ema_50 < 0.01
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: near EMA50 from below in uptrend with volume
        long_signal = near_ema50 and price_close > ema_50 and uptrend and vol_confirmed
        
        # Short: near EMA50 from above in downtrend with volume
        short_signal = near_ema50 and price_close < ema_50 and downtrend and vol_confirmed
        
        # Exit when price moves 2% away from EMA50 or trend changes
        exit_long = position == 1 and (price_close < ema_50 * 0.98 or price_close > ema_50 * 1.02 or not uptrend)
        exit_short = position == -1 and (price_close > ema_50 * 1.02 or price_close < ema_50 * 0.98 or not downtrend)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals