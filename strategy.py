#!/usr/bin/env python3
"""
4h RSI Divergence with Volume Confirmation and EMA Trend Filter
Hypothesis: Bullish/bearish RSI divergence on 4h, confirmed by volume > 1.5x EMA volume and price > EMA34, captures reversals in both bull and bear markets. RSI divergence is a leading indicator of trend exhaustion, effective in ranging and trending conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI calculation (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # EMA34 for trend filter
    ema34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ratio = volume / vol_ema
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Warmup for RSI and EMA
    
    for i in range(start_idx, n):
        if np.isnan(rsi[i]) or np.isnan(ema34[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        ema_val = ema34[i]
        vol_conf = vol_ratio[i] > 1.5
        
        # Detect RSI divergence: look back 3 bars for swing high/low
        if i >= 3:
            # Bullish divergence: price makes lower low, RSI makes higher low
            if (low[i] < low[i-1] < low[i-2] and 
                rsi[i] > rsi[i-1] > rsi[i-2] and
                price > ema_val and vol_conf):
                if position == 0:
                    signals[i] = 0.25
                    position = 1
                elif position == -1:
                    signals[i] = 0.25  # close short, go long
                    position = 1
            
            # Bearish divergence: price makes higher high, RSI makes lower high
            elif (high[i] > high[i-1] > high[i-2] and 
                  rsi[i] < rsi[i-1] < rsi[i-2] and
                  price < ema_val and vol_conf):
                if position == 0:
                    signals[i] = -0.25
                    position = -1
                elif position == 1:
                    signals[i] = -0.25  # close long, go short
                    position = -1
        
        # Exit conditions: RSI returns to neutral zone or trend breaks
        if position == 1:
            if rsi_val > 70 or price < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if rsi_val < 30 or price > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI_Divergence_Volume_EMA34"
timeframe = "4h"
leverage = 1.0