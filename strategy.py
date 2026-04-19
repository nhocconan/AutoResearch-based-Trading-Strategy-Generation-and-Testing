#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Supertrend with 1-week RSI filter and volume confirmation.
# Long when: Supertrend turns bullish, weekly RSI < 40 (oversold), volume > 1.5x 20-day average
# Short when: Supertrend turns bearish, weekly RSI > 60 (overbought), volume > 1.5x 20-day average
# Exit when Supertrend reverses.
# Supertrend captures trend direction, weekly RSI filters momentum extremes, volume confirms conviction.
# Target: 10-20 trades/year per symbol. Works in bull (buy trend pullbacks) and bear (sell rallies).
name = "1d_Supertrend_WeeklyRSI_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-week data for RSI filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly RSI (14-period)
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w = np.where((avg_gain == 0) & (avg_loss == 0), 50, rsi_1w)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Calculate Supertrend on daily data
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Basic Upper and Lower Bands
    basic_ub = (high + low) / 2 + multiplier * atr
    basic_lb = (high + low) / 2 - multiplier * atr
    
    # Final Upper and Lower Bands
    final_ub = np.zeros(n)
    final_lb = np.zeros(n)
    final_ub[0] = basic_ub[0]
    final_lb[0] = basic_lb[0]
    
    for i in range(1, n):
        final_ub[i] = basic_ub[i] if (basic_ub[i] < final_ub[i-1] or close[i-1] > final_ub[i-1]) else final_ub[i-1]
        final_lb[i] = basic_lb[i] if (basic_lb[i] > final_lb[i-1] or close[i-1] < final_lb[i-1]) else final_lb[i-1]
    
    # Supertrend
    supertrend = np.zeros(n)
    supertrend[0] = final_lb[0]
    for i in range(1, n):
        if supertrend[i-1] == final_ub[i-1]:
            supertrend[i] = final_ub[i] if close[i] <= final_ub[i] else final_lb[i]
        else:
            supertrend[i] = final_lb[i] if close[i] >= final_lb[i] else final_ub[i]
    
    # 20-day volume average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30)  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(supertrend[i]) or np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        st = supertrend[i]
        rsi = rsi_1w_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: Supertrend bullish (price > supertrend), weekly RSI < 40, volume spike
            if (price > st and rsi < 40 and vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: Supertrend bearish (price < supertrend), weekly RSI > 60, volume spike
            elif (price < st and rsi > 60 and vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Supertrend turns bearish (price < supertrend)
            if price < st:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Supertrend turns bullish (price > supertrend)
            if price > st:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals