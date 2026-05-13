#!/usr/bin/env python3
"""
4H_RSI_Divergence_Trend_Filter
Hypothesis: RSI divergence (bullish/bearish) on 4h chart with 12h trend filter (price > EMA50 for long, price < EMA50 for short) and volume confirmation (volume > 1.5x 20-period average). Designed to capture reversal points in both bull and bear markets with low trade frequency (<50/year) by requiring confluence of divergence, trend, and volume.
"""

name = "4H_RSI_Divergence_Trend_Filter"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate RSI(14) on 4h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    cooldown = 0  # cooldown counter to prevent immediate re-entry
    
    for i in range(30, n):
        # Decrease cooldown if active
        if cooldown > 0:
            cooldown -= 1
        
        if position == 0 and cooldown == 0:
            # Bullish RSI divergence: price makes lower low, RSI makes higher low
            bullish_div = False
            if i >= 3:
                # Look back 3 bars for lower low in price and higher low in RSI
                if low[i] < low[i-1] and low[i] < low[i-2] and low[i] < low[i-3]:
                    if rsi[i] > rsi[i-1] and rsi[i] > rsi[i-2] and rsi[i] > rsi[i-3]:
                        bullish_div = True
            
            # Bearish RSI divergence: price makes higher high, RSI makes lower high
            bearish_div = False
            if i >= 3:
                # Look back 3 bars for higher high in price and lower high in RSI
                if high[i] > high[i-1] and high[i] > high[i-2] and high[i] > high[i-3]:
                    if rsi[i] < rsi[i-1] and rsi[i] < rsi[i-2] and rsi[i] < rsi[i-3]:
                        bearish_div = True
            
            # LONG: Bullish divergence with uptrend and volume confirmation
            if bullish_div and ema_50_12h_aligned[i] > 0 and not np.isnan(ema_50_12h_aligned[i]) and \
               close[i] > ema_50_12h_aligned[i] and volume_confirmed[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish divergence with downtrend and volume confirmation
            elif bearish_div and ema_50_12h_aligned[i] > 0 and not np.isnan(ema_50_12h_aligned[i]) and \
                 close[i] < ema_50_12h_aligned[i] and volume_confirmed[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below EMA50 or bearish divergence
            if ema_50_12h_aligned[i] > 0 and not np.isnan(ema_50_12h_aligned[i]) and close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                cooldown = 3
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above EMA50 or bullish divergence
            if ema_50_12h_aligned[i] > 0 and not np.isnan(ema_50_12h_aligned[i]) and close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                cooldown = 3
            else:
                signals[i] = -0.25
    
    return signals