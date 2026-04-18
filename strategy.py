#!/usr/bin/env python3
"""
1h RSI Divergence with 4h Trend and Volume Filter
Hypothesis: RSI divergence on 1h combined with 4h EMA trend filter captures
mean-reversion opportunities in range-bound markets while avoiding trend traps.
Volume filter ensures institutional participation. Works in both bull and bear
markets by only taking divergence signals aligned with higher timeframe trend.
Target: 15-35 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # 4h EMA34 for trend filter
    ema34_4h = pd.Series(df_4h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Calculate RSI(14) on 1h
    delta = pd.Series(close).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: current volume > 1.5x 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for RSI
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        trend = ema34_4h_aligned[i]
        rsi_val = rsi[i]
        vol_ok = vol_filter[i]
        
        # Check for bullish RSI divergence (price makes lower low, RSI makes higher low)
        bullish_div = False
        if i >= 20:  # Need lookback for divergence
            # Find recent low in price
            price_low_idx = np.argmin(low[i-20:i]) + i - 20
            price_low_prev_idx = np.argmin(low[i-40:i-20]) + i - 40
            if (low[price_low_idx] < low[price_low_prev_idx] and 
                rsi[price_low_idx] > rsi[price_low_prev_idx]):
                bullish_div = True
        
        # Check for bearish RSI divergence (price makes higher high, RSI makes lower high)
        bearish_div = False
        if i >= 20:
            # Find recent high in price
            price_high_idx = np.argmax(high[i-20:i]) + i - 20
            price_high_prev_idx = np.argmax(high[i-40:i-20]) + i - 40
            if (high[price_high_idx] > high[price_high_prev_idx] and 
                rsi[price_high_idx] < rsi[price_high_prev_idx]):
                bearish_div = True
        
        if position == 0:
            # Long: bullish divergence in uptrend (4h EMA up) with volume
            if bullish_div and price > trend and vol_ok:
                signals[i] = 0.20
                position = 1
            # Short: bearish divergence in downtrend (4h EMA down) with volume
            elif bearish_div and price < trend and vol_ok:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: RSI overbought or trend breaks down
            if rsi_val > 70 or price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: RSI oversold or trend breaks up
            if rsi_val < 30 or price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI_Divergence_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0