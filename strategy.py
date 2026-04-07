#!/usr/bin/env python3
"""
6H Bollinger Band Width + RSI Mean Reversion with 12h Trend Filter
Hypothesis: In ranging markets (low BBW), price tends to revert from RSI extremes.
In trending markets (high BBW), we follow the 12h EMA trend.
This dual-regime approach works in both bull and bear markets by adapting to volatility regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_bbw_rsi_meanrev_12h_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Bollinger Bands (20, 2) ===
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_middle = sma20
    bb_width = (bb_upper - bb_lower) / (bb_middle + 1e-10)
    
    # === RSI (14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === 12h EMA trend filter ===
    df_12h = get_htf_data(prices, '12h')
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Regime thresholds
    bbw_ma = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    low_vol_threshold = bbw_ma * 0.5   # Below this = ranging market
    high_vol_threshold = bbw_ma * 2.0  # Above this = trending market
    
    for i in range(50, n):
        if (np.isnan(bb_width[i]) or np.isnan(rsi[i]) or 
            np.isnan(bbw_ma[i]) or np.isnan(ema_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI crosses above 60 or price touches upper BB
            if rsi[i] > 60 or close[i] >= bb_upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI crosses below 40 or price touches lower BB
            if rsi[i] < 40 or close[i] <= bb_lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Determine regime based on Bollinger Band Width
            if bb_width[i] < low_vol_threshold[i]:
                # Ranging market: mean reversion from RSI extremes
                if rsi[i] < 30 and close[i] > bb_lower[i]:
                    # Oversold and above lower BB -> long
                    position = 1
                    signals[i] = 0.25
                elif rsi[i] > 70 and close[i] < bb_upper[i]:
                    # Overbought and below upper BB -> short
                    position = -1
                    signals[i] = -0.25
            elif bb_width[i] > high_vol_threshold[i]:
                # Trending market: follow 12h EMA trend
                if ema_12h_aligned[i] > ema_12h_aligned[i-1] and close[i] > bb_middle[i]:
                    # Uptrend and above middle BB -> long
                    position = 1
                    signals[i] = 0.25
                elif ema_12h_aligned[i] < ema_12h_aligned[i-1] and close[i] < bb_middle[i]:
                    # Downtrend and below middle BB -> short
                    position = -1
                    signals[i] = -0.25
    
    return signals