#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy combining Bollinger Band squeeze with RSI momentum and volume filter
# Bollinger Band width contraction indicates low volatility and potential breakout
# RSI > 60 for longs and < 40 for shorts provides momentum confirmation
# Volume > 1.5x 20-period average confirms breakout strength
# Works in bull/bear markets: captures breakouts from low volatility regimes
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "6h_BB_Squeeze_RSI_Momentum_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) - calculate once
    close_series = pd.Series(close)
    bb_ma = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band width percentile (20-period lookback) for squeeze detection
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # RSI (14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    avg_gain = gain_series.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = loss_series.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(bb_width_percentile[i]) or np.isnan(rsi[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: BB squeeze (low volatility) + RSI bullish + volume breakout
            if (bb_width_percentile[i] <= 20 and  # Bollinger Band squeeze
                rsi[i] > 60 and                   # Bullish momentum
                close[i] > bb_upper[i] and        # Break above upper band
                volume_filter[i]):                # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short entry: BB squeeze + RSI bearish + volume breakdown
            elif (bb_width_percentile[i] <= 20 and  # Bollinger Band squeeze
                  rsi[i] < 40 and                   # Bearish momentum
                  close[i] < bb_lower[i] and        # Break below lower band
                  volume_filter[i]):                # Volume confirmation
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI turns bearish or price returns to middle band
            if rsi[i] < 50 or close[i] < bb_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI turns bullish or price returns to middle band
            if rsi[i] > 50 or close[i] > bb_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals