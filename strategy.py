#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_4hour_Trend_Pullback_With_Volume_Confirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily EMA20 for trend filter
    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Daily ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    tr = np.maximum(high_1d - low_1d, 
                    np.maximum(np.abs(high_1d - np.roll(close_1d_arr, 1)), 
                               np.abs(low_1d - np.roll(close_1d_arr, 1))))
    tr[0] = high_1d[0] - low_1d[0]
    atr14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # 4-hour RSI(14) for pullback entry
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 4-hour volume filter: current volume > 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for RSI and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema20_1d_aligned[i]) or np.isnan(atr14_1d_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: pullback in uptrend (price above daily EMA20, RSI < 40, volume confirmation)
            long_cond = (close[i] > ema20_1d_aligned[i] and 
                        rsi[i] < 40 and 
                        volume[i] > vol_ma20[i])
            
            # Short: pullback in downtrend (price below daily EMA20, RSI > 60, volume confirmation)
            short_cond = (close[i] < ema20_1d_aligned[i] and 
                         rsi[i] > 60 and 
                         volume[i] > vol_ma20[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend reversal or overbought
            if close[i] < ema20_1d_aligned[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend reversal or oversold
            if close[i] > ema20_1d_aligned[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Trend-following with pullback entries on 4h timeframe using daily EMA20 as trend filter.
# Enters long when price is above daily EMA20 (uptrend) and RSI < 40 (pullback) with volume confirmation.
# Enters short when price is below daily EMA20 (downtrend) and RSI > 60 (pullback bounce) with volume confirmation.
# Exits when trend reverses (price crosses daily EMA20) or RSI reaches extreme levels.
# Designed to work in both bull and bear markets by following the higher timeframe trend.
# Uses discrete sizing (0.25) to minimize churn and targets ~25-50 trades per year to avoid fee drag.