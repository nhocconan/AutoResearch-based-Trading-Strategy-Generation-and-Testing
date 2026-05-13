#!/usr/bin/env python3
"""
4h_RSI_Divergence_Pullback_Trend
Hypothesis: RSI divergence at support/resistance levels combined with EMA trend and volume confirmation captures high-probability pullbacks in trending markets. Works in bull/bear by using EMA trend filter and only taking long in uptrend, short in downtrend.
"""

name = "4h_RSI_Divergence_Pullback_Trend"
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
    
    # Get daily data for trend filter and key levels (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA50 on daily close for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate RSI(14) on 4h closes
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(0).values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # Check for bullish RSI divergence: price makes lower low, RSI makes higher low
            bullish_div = False
            if i >= 20:  # Need enough history for divergence check
                # Look back up to 20 bars for a swing low
                lookback = min(20, i)
                price_lows = []
                rsi_lows = []
                for j in range(i - lookback, i + 1):
                    if j >= 2 and j < n - 2:
                        # Simple swing low detection: lower than neighbors
                        if low[j] <= low[j-1] and low[j] <= low[j+1]:
                            price_lows.append((j, low[j]))
                            rsi_lows.append((j, rsi[j]))
                
                # Check for bullish divergence: at least 2 lows where price lower but RSI higher
                if len(price_lows) >= 2:
                    # Sort by price (lowest first)
                    price_lows_sorted = sorted(price_lows, key=lambda x: x[1])
                    rsi_lows_sorted = sorted(rsi_lows, key=lambda x: x[1])
                    # Check if lowest price corresponds to higher RSI than a previous low
                    if price_lows_sorted[0][1] < price_lows_sorted[-1][1] and \
                       rsi_lows_sorted[0][1] > rsi_lows_sorted[-1][1]:
                        bullish_div = True
            
            # Check for bearish RSI divergence: price makes higher high, RSI makes lower high
            bearish_div = False
            if i >= 20:
                lookback = min(20, i)
                price_highs = []
                rsi_highs = []
                for j in range(i - lookback, i + 1):
                    if j >= 2 and j < n - 2:
                        # Simple swing high detection: higher than neighbors
                        if high[j] >= high[j-1] and high[j] >= high[j+1]:
                            price_highs.append((j, high[j]))
                            rsi_highs.append((j, rsi[j]))
                
                if len(price_highs) >= 2:
                    price_highs_sorted = sorted(price_highs, key=lambda x: x[1], reverse=True)
                    rsi_highs_sorted = sorted(rsi_highs, key=lambda x: x[1], reverse=True)
                    if price_highs_sorted[0][1] > price_highs_sorted[-1][1] and \
                       rsi_highs_sorted[0][1] < rsi_highs_sorted[-1][1]:
                        bearish_div = True
            
            # LONG: bullish divergence + price above daily EMA50 (uptrend) + volume spike
            if bullish_div and close[i] > trend_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: bearish divergence + price below daily EMA50 (downtrend) + volume spike
            elif bearish_div and close[i] < trend_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: bearish divergence or price breaks below EMA50
            exit_signal = False
            if i >= 20:
                lookback = min(20, i)
                price_highs = []
                rsi_highs = []
                for j in range(i - lookback, i + 1):
                    if j >= 2 and j < n - 2:
                        if high[j] >= high[j-1] and high[j] >= high[j+1]:
                            price_highs.append((j, high[j]))
                            rsi_highs.append((j, rsi[j]))
                if len(price_highs) >= 2:
                    price_highs_sorted = sorted(price_highs, key=lambda x: x[1], reverse=True)
                    rsi_highs_sorted = sorted(rsi_highs, key=lambda x: x[1], reverse=True)
                    if price_highs_sorted[0][1] > price_highs_sorted[-1][1] and \
                       rsi_highs_sorted[0][1] < rsi_highs_sorted[-1][1]:
                        exit_signal = True
            if exit_signal or close[i] < trend_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: bullish divergence or price breaks above EMA50
            exit_signal = False
            if i >= 20:
                lookback = min(20, i)
                price_lows = []
                rsi_lows = []
                for j in range(i - lookback, i + 1):
                    if j >= 2 and j < n - 2:
                        if low[j] <= low[j-1] and low[j] <= low[j+1]:
                            price_lows.append((j, low[j]))
                            rsi_lows.append((j, rsi[j]))
                if len(price_lows) >= 2:
                    price_lows_sorted = sorted(price_lows, key=lambda x: x[1])
                    rsi_lows_sorted = sorted(rsi_lows, key=lambda x: x[1])
                    if price_lows_sorted[0][1] < price_lows_sorted[-1][1] and \
                       rsi_lows_sorted[0][1] > rsi_lows_sorted[-1][1]:
                        exit_signal = True
            if exit_signal or close[i] > trend_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals