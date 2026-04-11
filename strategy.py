#!/usr/bin/env python3
"""
4h_1d_keltner_channel_breakout_volume_v1
Strategy: 4h Keltner Channel breakout with volume confirmation and 1d trend filter
Timeframe: 4h
Leverage: 1.0
Hypothesis: Uses Keltner Channel (EMA20 + 2*ATR(10)) breakouts for trend continuation. 
Volume confirmation (>1.5x average volume) filters false breakouts. 
1d EMA50 trend filter ensures alignment with higher timeframe trend. 
Designed to work in both bull and bear markets by capturing strong momentum moves 
while avoiding choppy conditions. Target: 20-50 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_keltner_channel_breakout_volume_v1"
timeframe = "4h"
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
    
    # 4h EMA20 for Keltner Channel middle line
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 4h ATR(10) for Keltner Channel width
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner Channel bands
    kc_upper = ema_20 + 2 * atr_10
    kc_lower = ema_20 - 2 * atr_10
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_20[i]) or np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below 1d EMA50
        uptrend = price_close > ema_50_1d_aligned[i]
        downtrend = price_close < ema_50_1d_aligned[i]
        
        # Breakout conditions
        breakout_up = price_close > kc_upper[i]
        breakout_down = price_close < kc_lower[i]
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: upward breakout with volume in uptrend
        long_signal = breakout_up and vol_confirmed and uptrend
        
        # Short: downward breakout with volume in downtrend
        short_signal = breakout_down and vol_confirmed and downtrend
        
        # Exit when price returns to the middle line (EMA20)
        exit_long = position == 1 and price_close < ema_20[i]
        exit_short = position == -1 and price_close > ema_20[i]
        
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