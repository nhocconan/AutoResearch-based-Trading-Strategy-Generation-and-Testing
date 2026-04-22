#!/usr/bin/env python3
"""
Hypothesis: 6-hour Bollinger Band Width regime with 1-week RSI trend filter.
In low volatility (BB Width < 20th percentile), mean revert at Bollinger Bands.
In high volatility (BB Width > 80th percentile), follow 1-week RSI trend (RSI>50 long, <50 short).
BB Width regime identifies market state; 1-week RSI provides trend filter for breakouts.
Works in bull markets (trend following breakouts) and bear markets (mean reversion in ranges).
"""

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
    
    # Bollinger Bands (20, 2) on 6h
    basis = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    dev = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper = basis + 2.0 * dev
    lower = basis - 2.0 * dev
    bb_width = (upper - lower) / basis
    
    # Percentile rank of BB Width (50-period lookback)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == 50 else np.nan, raw=False
    ).values
    
    # Load 1-week RSI for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bb_width_percentile[i]) or np.isnan(rsi_1w_aligned[i]) or
            np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(basis[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Entry logic based on regime
            if bb_width_percentile[i] < 0.20:  # Low volatility - mean revert
                if close[i] < lower[i]:  # Oversold - go long
                    signals[i] = 0.25
                    position = 1
                elif close[i] > upper[i]:  # Overbought - go short
                    signals[i] = -0.25
                    position = -1
            elif bb_width_percentile[i] > 0.80:  # High volatility - follow trend
                if rsi_1w_aligned[i] > 50:  # Uptrend - go long
                    signals[i] = 0.25
                    position = 1
                elif rsi_1w_aligned[i] < 50:  # Downtrend - go short
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses above mean (basis) or volatility regime changes
                if close[i] > basis[i] or bb_width_percentile[i] > 0.80:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses below mean (basis) or volatility regime changes
                if close[i] < basis[i] or bb_width_percentile[i] > 0.80:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_BBWidth_Regime_1wRSI_Trend"
timeframe = "6h"
leverage = 1.0