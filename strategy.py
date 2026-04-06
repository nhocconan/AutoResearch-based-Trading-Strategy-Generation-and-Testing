#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h RSI with 1d Bollinger Bands for regime detection.
# In low volatility (bandwidth < 50th percentile), mean revert at RSI extremes (30/70).
# In high volatility (bandwidth >= 50th percentile), trend follow RSI crosses (50).
# Uses 1d BBands to filter regime, avoids whipsaw in ranging markets.
# Designed for 6h timeframe with ~100-200 trades over 4 years (25-50/year).
# Works in bull markets (trend follow in high vol) and bear markets (mean revert in low vol).

name = "6h_rsi_1d_bbands_regime_v1"
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
    
    # 1d data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d Bollinger Bands (20, 2)
    close_1d = df_1d['close'].values
    bb_middle = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower  # Bandwidth
    
    # Percentile of bandwidth (50th = median)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Align Bollinger width percentile to 6h
    bb_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    
    # 6h RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(bb_percentile_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Regime: low volatility = mean revert, high volatility = trend follow
        low_vol = bb_percentile_aligned[i] < 50  # Below median bandwidth
        
        if position == 1:  # long position
            # Exit conditions
            if low_vol:
                # Mean revert: exit at RSI 70 (overbought)
                if rsi[i] >= 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # Trend follow: exit at RSI 50 (mean reversion)
                if rsi[i] <= 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:  # short position
            # Exit conditions
            if low_vol:
                # Mean revert: exit at RSI 30 (oversold)
                if rsi[i] <= 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # Trend follow: exit at RSI 50 (mean reversion)
                if rsi[i] >= 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # Look for entries
            if low_vol:
                # Mean reversion regime: buy at RSI 30, sell at RSI 70
                if rsi[i] <= 30:
                    signals[i] = 0.25
                    position = 1
                elif rsi[i] >= 70:
                    signals[i] = -0.25
                    position = -1
            else:
                # Trend following regime: buy at RSI > 50, sell at RSI < 50
                if rsi[i] > 50:
                    signals[i] = 0.25
                    position = 1
                elif rsi[i] < 50:
                    signals[i] = -0.25
                    position = -1
    
    return signals