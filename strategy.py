# 2024-06-08 19:32:32
#!/usr/bin/env python3
"""
4h RSI(14) Extreme + 1-day Volume Confirmation
Hypothesis: RSI extremes (overbought/oversold) signal mean reversion opportunities.
Volume confirmation ensures institutional participation. 1-day trend filter prevents
trading against strong trends. Works in both bull and bear markets by fading extremes
with volume confirmation.
"""

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
    
    # Get 1d data for trend and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d average volume for confirmation
    avg_vol_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(avg_vol_1d_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # Conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        vol_confirm = volume[i] > avg_vol_1d_aligned[i] * 1.5
        price_above_ema = close[i] > ema50_1d_aligned[i]
        price_below_ema = close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # Enter long on RSI oversold + volume confirmation + price above EMA (not in strong downtrend)
            if rsi_oversold and vol_confirm and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Enter short on RSI overbought + volume confirmation + price below EMA (not in strong uptrend)
            elif rsi_overbought and vol_confirm and price_below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long on RSI recovery or price below EMA
            if rsi[i] > 50 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short on RSI decline or price above EMA
            if rsi[i] < 50 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI_Extreme_Volume_Confirmation_1dTrend"
timeframe = "4h"
leverage = 1.0