#!/usr/bin/env python3
# 12h_Keltner_Channel_Breakout_Trend_Volume
# Hypothesis: Keltner Channel breakout with 1d trend filter (EMA34) and volume confirmation
# Keltner Channel uses ATR-based bands which adapt to volatility, providing dynamic support/resistance
# Trend filter prevents counter-trend trades in strong trends, improving win rate in both bull and bear markets
# Volume confirmation ensures breakouts have institutional participation
# Target: 50-150 trades over 4 years (12-37/year) with position size 0.25

name = "12h_Keltner_Channel_Breakout_Trend_Volume"
timeframe = "12h"
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
    
    # Load daily data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # ATR(14) for Keltner Channel
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # First period has no previous close
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # EMA20 for Keltner Channel middle line
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel: Upper = EMA20 + 2*ATR, Lower = EMA20 - 2*ATR
    kc_upper = ema_20 + 2.0 * atr
    kc_lower = ema_20 - 2.0 * atr
    
    # Volume confirmation: volume > 1.5 * average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_12h[i]) or np.isnan(ema_20[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(kc_upper[i]) or np.isnan(kc_lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: only take longs in daily uptrend, shorts in daily downtrend
        daily_uptrend = close[i] > ema_34_12h[i]
        daily_downtrend = close[i] < ema_34_12h[i]
        
        if position == 0:
            # Long: price breaks above KC upper + daily uptrend + volume confirmation
            if close[i] > kc_upper[i] and daily_uptrend and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below KC lower + daily downtrend + volume confirmation
            elif close[i] < kc_lower[i] and daily_downtrend and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price closes below KC middle OR trend reversal
            if close[i] < ema_20[i] or not daily_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price closes above KC middle OR trend reversal
            if close[i] > ema_20[i] or not daily_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals