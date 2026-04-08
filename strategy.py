#!/usr/bin/env python3
# 12h_1d_ema_volume_squeeze_v1
# Hypothesis: In low-volatility regimes (Bollinger Band width < 50th percentile), price breaking above/below 20-period EMA with volume > 1.5x 20-period average triggers mean-reversion trades.
# Long when price crosses above EMA20 with volume surge in low volatility; short when price crosses below EMA20 with volume surge in low volatility.
# Uses 1d EMA20 trend filter to avoid counter-trend trades. Designed for 12-37 trades/year on 12h timeframe.
# Works in ranging markets via mean reversion and avoids trending markets via volatility regime filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_ema_volume_squeeze_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h EMA(20) for entry signal
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 12h Bollinger Bands for volatility regime (20, 2)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    bb_width = (upper - lower) / sma20  # Normalized width
    
    # 12h volume MA(20) for volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(20) for trend filter
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Bollinger Band width percentile (lookback 50 periods) for regime filter
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema20[i]) or np.isnan(sma20[i]) or np.isnan(std20[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(ema20_1d_aligned[i]) or 
            np.isnan(bb_width_percentile[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Low volatility regime: BB width below 50th percentile
        low_volatility = bb_width_percentile[i] < 0.5
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price crosses below EMA20 or volatility increases
            if close[i] < ema20[i] or not low_volatility:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above EMA20 or volatility increases
            if close[i] > ema20[i] or not low_volatility:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price crosses above EMA20 with volume surge in low volatility and uptrend (1d EMA)
            if (close[i] > ema20[i] and close[i-1] <= ema20[i-1] and  # Cross above
                vol_surge and low_volatility and close[i] > ema20_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price crosses below EMA20 with volume surge in low volatility and downtrend (1d EMA)
            elif (close[i] < ema20[i] and close[i-1] >= ema20[i-1] and  # Cross below
                  vol_surge and low_volatility and close[i] < ema20_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals