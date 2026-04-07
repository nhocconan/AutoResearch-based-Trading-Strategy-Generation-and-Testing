#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze + 1w Trend Filter + Volume Spike
# In ranging markets (low volatility), Bollinger Bands contract (squeeze).
# When bands expand after squeeze, price tends to break out in direction of higher timeframe trend.
# Uses 1w EMA200 for trend filter to avoid counter-trend trades.
# Volume spike confirms breakout validity.
# Designed for low frequency (~20-40 trades/year) with clear entry/exit rules.
# Works in bull/bear via trend filter: only trade breaks in direction of 1w trend.

name = "6h_bollinger_squeeze_1w_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w trend filter (EMA200)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Bollinger Bands (20, 2) on 6h
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma + (std * bb_std)
    lower = sma - (std * bb_std)
    
    # Bollinger Band Width (normalized by SMA) for squeeze detection
    bb_width = (upper - lower) / sma
    # Squeeze: BB width below 20-period percentile (20th percentile = low volatility)
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=30).rank(pct=True).values
    squeeze = bb_width_percentile < 0.2  # In squeeze when width is in lowest 20%
    
    # Volume spike confirmation (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(sma[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(squeeze[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from 1w EMA200
        uptrend = close[i] > ema_200_1w_aligned[i]
        downtrend = close[i] < ema_200_1w_aligned[i]
        
        # Exit conditions: reverse signal or re-entry into squeeze (mean reversion)
        if position == 1:  # Long position
            # Exit on breakdown below lower band OR re-entry into squeeze
            if (close[i] < lower[i]) or squeeze[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit on break above upper band OR re-entry into squeeze
            if (close[i] > upper[i]) or squeeze[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Only trade breakouts after squeeze, in direction of 1w trend
            if squeeze[i-1] and not squeeze[i]:  # Squeeze just released (expanding volatility)
                # Bullish breakout: price breaks above upper band with volume and uptrend
                if (close[i] > upper[i] * 1.001) and uptrend and vol_spike[i]:
                    position = 1
                    signals[i] = 0.25
                # Bearish breakout: price breaks below lower band with volume and downtrend
                elif (close[i] < lower[i] * 0.999) and downtrend and vol_spike[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals