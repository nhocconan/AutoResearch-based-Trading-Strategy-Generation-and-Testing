#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band mean reversion with weekly trend filter
# In ranging markets (high Bollinger Bandwidth percentile), price tends to revert to the mean (BB middle).
# In trending markets (low BB width percentile), we follow the weekly trend.
# This adapts to both bull and bear regimes by using volatility regime detection.
# Target: 15-25 trades/year to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter and regime detection
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema20_1w = close_1w.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Daily Bollinger Bands (20, 2)
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Bollinger Bandwidth for regime detection (normalized by middle band)
    bb_width = (bb_upper - bb_lower) / bb_middle
    # Percentile of BB width over 60 days to detect regime
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=60, min_periods=30).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(bb_middle[i]) or 
            np.isnan(bb_width_percentile[i])):
            signals[i] = 0.0
            continue
        
        # Regime detection: high BB width = ranging market (mean revert)
        # Low BB width = trending market (follow trend)
        is_ranging = bb_width_percentile[i] > 0.6  # Top 40% = ranging
        
        if position == 0:
            if is_ranging:
                # In ranging market: mean reversion at Bollinger Bands
                if close[i] <= bb_lower[i]:  # Oversold
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= bb_upper[i]:  # Overbought
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # In trending market: follow weekly trend
                if close[i] > ema20_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < ema20_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses above middle (mean reversion) or below weekly EMA (trend change)
            if is_ranging:
                if close[i] >= bb_middle[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                if close[i] < ema20_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses below middle (mean reversion) or above weekly EMA (trend change)
            if is_ranging:
                if close[i] <= bb_middle[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                if close[i] > ema20_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_BB_MeanReversion_Trend_Adaptive_v1"
timeframe = "1d"
leverage = 1.0