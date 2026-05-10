#!/usr/bin/env python3
# 4H_RSI_Divergence_With_12hTrend
# Hypothesis: RSI divergence signals filtered by 12h trend and volume confirmation.
# Bullish divergence (price low, RSI higher low) + 12h uptrend + volume spike -> long
# Bearish divergence (price high, RSI lower high) + 12h downtrend + volume spike -> short
# Exit when RSI returns to neutral zone (40-60) or opposite divergence occurs.
# Designed for low trade frequency (<30/year) to avoid fee drag, works in bull/bear by following 12h trend.

name = "4H_RSI_Divergence_With_12hTrend"
timeframe = "4h"
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
    
    # RSI(14) calculation
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # 12h trend (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_12h_up = close_12h > ema50_12h
    trend_12h_down = close_12h < ema50_12h
    
    # Align 12h trend to 4h
    trend_12h_up_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_up.astype(float))
    trend_12h_down_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_down.astype(float))
    
    # Volume average (20-period)
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(trend_12h_up_aligned[i]) or np.isnan(trend_12h_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 2.0
        
        trend_up = trend_12h_up_aligned[i] > 0.5
        trend_down = trend_12h_down_aligned[i] > 0.5
        
        # Check for RSI divergence (need at least 3 bars back)
        if i >= 3:
            # Bullish divergence: price makes lower low, RSI makes higher low
            if (low[i] < low[i-2] and low[i-2] < low[i-4] and 
                rsi[i] > rsi[i-2] and rsi[i-2] > rsi[i-4]):
                bullish_div = True
            else:
                bullish_div = False
            
            # Bearish divergence: price makes higher high, RSI makes lower high
            if (high[i] > high[i-2] and high[i-2] > high[i-4] and 
                rsi[i] < rsi[i-2] and rsi[i-2] < rsi[i-4]):
                bearish_div = True
            else:
                bearish_div = False
        else:
            bullish_div = False
            bearish_div = False
        
        # RSI neutral zone (exit condition)
        rsi_neutral = 40 <= rsi[i] <= 60
        
        if position == 0:
            # Enter long: bullish divergence + 12h uptrend + volume
            if bullish_div and trend_up and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish divergence + 12h downtrend + volume
            elif bearish_div and trend_down and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral or bearish divergence
            if rsi_neutral or bearish_div:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral or bullish divergence
            if rsi_neutral or bullish_div:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals