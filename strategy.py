#!/usr/bin/env python3
name = "6h_RSI_Trend_Range_Adaptive"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for trend filter and chop regime
    df_1d = get_htf_data(prices, '1d')
    
    # Daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Daily ATR(14) for chop regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    tr1 = np.maximum(high_1d - low_1d, 
                     np.absolute(high_1d - np.roll(close_1d_arr, 1)),
                     np.absolute(low_1d - np.roll(close_1d_arr, 1)))
    tr1[0] = high_1d[0] - low_1d[0]
    atr14_1d = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # Daily range for chop calculation
    daily_range = high_1d - low_1d
    daily_range_aligned = align_htf_to_ltf(prices, df_1d, daily_range)
    
    # Chop ratio: ATR(14) / daily range (lower = trending, higher = ranging)
    chop_ratio = atr14_1d_aligned / daily_range_aligned
    
    # 6h RSI(14) for entry signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(chop_ratio[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Regime detection: chop_ratio < 0.6 = trending, chop_ratio >= 0.6 = ranging
        is_trending = chop_ratio[i] < 0.6
        
        if position == 0:
            if is_trending:
                # Trend following: RSI pullback in trend direction
                if rsi[i] < 40 and close[i] > ema50_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                elif rsi[i] > 60 and close[i] < ema50_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                # Mean reversion: RSI extremes
                if rsi[i] < 30:
                    signals[i] = 0.25
                    position = 1
                elif rsi[i] > 70:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit conditions
            if is_trending:
                # Exit trend: RSI overbought or price below EMA
                if rsi[i] > 70 or close[i] < ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # Exit mean reversion: RSI returns to neutral
                if rsi[i] > 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Exit conditions
            if is_trending:
                # Exit trend: RSI oversold or price above EMA
                if rsi[i] < 30 or close[i] > ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # Exit mean reversion: RSI returns to neutral
                if rsi[i] < 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals