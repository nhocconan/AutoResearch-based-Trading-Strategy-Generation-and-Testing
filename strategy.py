#!/usr/bin/env python3
"""
4h_1d_Momentum_Reversal_v2
Hypothesis: In both bull and bear markets, price reverses sharply after touching 1-day Bollinger Bands with RSI divergence and volume spike.
Long when price touches lower BB with RSI<30 and volume spike; short when touches upper BB with RSI>70.
Use 4h for entry timing and 1d for regime (BB/RSI). Target 20-40 trades/year.
Works in bull (momentum continuation) and bear (mean reversion at extremes).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Momentum_Reversal_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for BB, RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # === 1D BOLLINGER BANDS (20, 2.0) ===
    sma20 = pd.Series(daily_close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(daily_close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2.0 * std20
    lower_bb = sma20 - 2.0 * std20
    
    upper_bb_4h = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_4h = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # === 1D RSI(14) ===
    delta = pd.Series(daily_close).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_4h = align_htf_to_ltf(prices, df_1d, rsi)
    
    # === VOLUME SPIKE (2x) ===
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        vol_sum = np.sum(volume[:20])
        vol_ma[19] = vol_sum / 20
        for i in range(20, n):
            vol_sum = vol_sum - volume[i-20] + volume[i]
            vol_ma[i] = vol_sum / 20
    vol_spike = volume > (vol_ma * 2.0)  # Volume > 2x average
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any data invalid
        if (np.isnan(upper_bb_4h[i]) or np.isnan(lower_bb_4h[i]) or 
            np.isnan(rsi_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Reversal signals at Bollinger Bands
        touch_lower = low[i] <= lower_bb_4h[i]
        touch_upper = high[i] >= upper_bb_4h[i]
        
        # RSI extremes for reversal confirmation
        rsi_oversold = rsi_4h[i] < 30
        rsi_overbought = rsi_4h[i] > 70
        
        # Entry conditions
        long_entry = touch_lower and rsi_oversold and vol_spike[i]
        short_entry = touch_upper and rsi_overbought and vol_spike[i]
        
        # Exit: reverse signal or price returns to SMA20 (middle BB)
        sma20_4h = align_htf_to_ltf(prices, df_1d, sma20)
        long_exit = not touch_lower or close[i] >= sma20_4h[i]
        short_exit = not touch_upper or close[i] <= sma20_4h[i]
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals