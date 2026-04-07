#!/usr/bin/env python3
"""
1d Bollinger Band Squeeze Breakout with Weekly Trend Filter
Long when price breaks above upper BB during low volatility (squeeze) and weekly trend is up
Short when price breaks below lower BB during low volatility and weekly trend is down
Exit when price returns to middle BB or volatility expands
Bollinger Squeeze captures low volatility breakouts which work in both bull and bear markets
Weekly trend filter ensures we trade with the higher timeframe momentum
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_bollinger_squeeze_breakout_weekly_trend_v1"
timeframe = "1d"
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
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean().values
    dev = close_s.rolling(window=20, min_periods=20).std().values
    upper_band = basis + 2.0 * dev
    lower_band = basis - 2.0 * dev
    
    # === Bollinger Band Width (for squeeze detection) ===
    bb_width = (upper_band - lower_band) / (basis + 1e-10)
    bb_width_ma = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    squeeze_condition = bb_width < 0.5 * bb_width_ma  # Low volatility squeeze
    
    # === Weekly Trend Filter (using 1h data as proxy for weekly trend) ===
    # Get weekly data (using 1h as closest available)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 50:
        return np.zeros(n)
    
    # Weekly EMA trend (using 50-period EMA on 1h data as proxy)
    close_1h = pd.Series(df_1h['close'].values)
    ema_50_1h = close_1h.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_50_1h)
    
    # Weekly trend direction
    weekly_uptrend = ema_50_1h_aligned > close  # Price above weekly EMA = uptrend
    weekly_downtrend = ema_50_1h_aligned < close  # Price below weekly EMA = downtrend
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        if np.isnan(basis[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(squeeze_condition[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to middle band or volatility expands (squeeze ends)
            if close[i] <= basis[i] or not squeeze_condition[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to middle band or volatility expands
            if close[i] >= basis[i] or not squeeze_condition[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need low volatility squeeze
            if not squeeze_condition[i]:
                signals[i] = 0.0
                continue
            
            # Entry: Bollinger Band breakout with weekly trend alignment
            if close[i] > upper_band[i] and weekly_uptrend[i]:
                # Break above upper band during uptrend -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < lower_band[i] and weekly_downtrend[i]:
                # Break below lower band during downtrend -> short
                position = -1
                signals[i] = -0.25
    
    return signals