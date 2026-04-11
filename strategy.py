#!/usr/bin/env python3
# 4h_1d_rsi_divergence_v1
# Strategy: 4h RSI divergence with 1d trend filter and volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: RSI divergences signal reversals; 1d EMA filters trend direction; volume confirms momentum.
# Long when bullish RSI divergence + price > 1d EMA50 + volume > 1.5x avg; short when bearish RSI divergence + price < 1d EMA50 + volume > 1.5x avg.
# Designed for low frequency (~25-35 trades/year) to minimize fee drag while capturing reversals in all market regimes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_rsi_divergence_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # RSI calculation (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi.iloc[i]) or 
            np.isnan(avg_volume.iloc[i]) or i < 30):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter from 1d EMA
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * avg_volume.iloc[i]
        
        # Bullish RSI divergence: price makes lower low, RSI makes higher low
        bullish_div = False
        if i >= 3:
            # Look for price making lower low over last 3 periods
            if low[i] < low[i-1] and low[i-1] < low[i-2]:
                # RSI making higher low over same period
                if rsi.iloc[i] > rsi.iloc[i-1] and rsi.iloc[i-1] > rsi.iloc[i-2]:
                    bullish_div = True
        
        # Bearish RSI divergence: price makes higher high, RSI makes lower high
        bearish_div = False
        if i >= 3:
            # Look for price making higher high over last 3 periods
            if high[i] > high[i-1] and high[i-1] > high[i-2]:
                # RSI making lower high over same period
                if rsi.iloc[i] < rsi.iloc[i-1] and rsi.iloc[i-1] < rsi.iloc[i-2]:
                    bearish_div = True
        
        # Entry conditions
        if bullish_div and uptrend and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_div and downtrend and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: trend reversal or loss of momentum
        elif position == 1 and (not uptrend or rsi.iloc[i] > 70):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not downtrend or rsi.iloc[i] < 30):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals