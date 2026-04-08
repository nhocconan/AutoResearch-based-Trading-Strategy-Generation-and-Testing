#!/usr/bin/env python3
# 1d_momentum_reversion_v1
# Hypothesis: On daily timeframe, capture mean reversion in ranging markets (Chop > 61.8) using RSI extremes, 
# and trend continuation in trending markets (Chop < 38.2) using price > EMA50. Uses 1-week EMA200 for regime filter.
# Works in bull/bear by adapting to market regime. Low trade frequency (~10-20/year) minimizes fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_momentum_reversion_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA200 for regime filter
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Daily EMA50 for trend following
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14)
    tr1 = high - low
    tr2 = np.abs(np.concatenate([[high[0]], high[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(np.concatenate([[low[0]], low[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10((atr * 14) / (max_high - min_low)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema_50[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_200_1w_aligned[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI < 40 (loss of momentum) OR price < EMA50 (trend broken) OR Chop > 61.8 (ranging - switch to mean reversion)
            if (rsi[i] < 40) or (close[i] < ema_50[i]) or (chop[i] > 61.8):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI > 60 (loss of momentum) OR price > EMA50 (trend broken) OR Chop > 61.8 (ranging - switch to mean reversion)
            if (rsi[i] > 60) or (close[i] > ema_50[i]) or (chop[i] > 61.8):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if chop[i] > 61.8:  # Ranging market - mean reversion
                # Long entry: RSI < 30 (oversold) AND price > EMA50 (avoid strong downtrend)
                if (rsi[i] < 30) and (close[i] > ema_50[i]):
                    position = 1
                    signals[i] = 0.25
                # Short entry: RSI > 70 (overbought) AND price < EMA50 (avoid strong uptrend)
                elif (rsi[i] > 70) and (close[i] < ema_50[i]):
                    position = -1
                    signals[i] = -0.25
            elif chop[i] < 38.2:  # Trending market - trend following
                # Long entry: price > EMA50 AND close > weekly EMA200 (bullish regime)
                if (close[i] > ema_50[i]) and (close[i] > ema_200_1w_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Short entry: price < EMA50 AND close < weekly EMA200 (bearish regime)
                elif (close[i] < ema_50[i]) and (close[i] < ema_200_1w_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals