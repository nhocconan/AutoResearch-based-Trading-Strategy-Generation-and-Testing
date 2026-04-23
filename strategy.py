#!/usr/bin/env python3
"""
Hypothesis: 1h Bollinger Band squeeze breakout with 4h EMA trend filter and volume confirmation.
In low volatility regimes (Bollinger Band Width at 20-period low), breakouts tend to trend.
Enter in direction of 4h EMA (50) when price breaks Bollinger Bands (20,2) with volume > 1.5x 20-period average.
Avoids whipsaw in high volatility and ranges. Designed for 1h timeframe to achieve 15-37 trades/year.
"""

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
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h Bollinger Bands (20,2)
    bb_period = 20
    bb_std = 2.0
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + (bb_std * bb_std_dev)
    lower_band = sma - (bb_std * bb_std_dev)
    bb_width = upper_band - lower_band
    
    # Calculate Bollinger Band Width percentile (lookback 50 periods) to identify squeeze
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    
    # Calculate volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(bb_period, 50, 20)  # need BB, EMA4h, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(sma[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(bb_width_percentile[i]) or np.isnan(vol_ma[i]) or np.isnan(ema_50_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: BB squeeze (width < 20th percentile) AND price breaks above upper band AND volume > 1.5x avg AND uptrend (price > 4h EMA50)
            if (bb_width_percentile[i] < 0.20 and 
                close[i] > upper_band[i] and 
                volume[i] > 1.5 * vol_ma[i] and 
                close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: BB squeeze (width < 20th percentile) AND price breaks below lower band AND volume > 1.5x avg AND downtrend (price < 4h EMA50)
            elif (bb_width_percentile[i] < 0.20 and 
                  close[i] < lower_band[i] and 
                  volume[i] > 1.5 * vol_ma[i] and 
                  close[i] < ema_50_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit: price returns to middle band (mean reversion) OR volatility expands (BB width > 80th percentile)
            exit_signal = False
            if position == 1:
                # Exit long when price <= middle band OR volatility expands significantly
                if close[i] <= sma[i] or bb_width_percentile[i] > 0.80:
                    exit_signal = True
            elif position == -1:
                # Exit short when price >= middle band OR volatility expands significantly
                if close[i] >= sma[i] or bb_width_percentile[i] > 0.80:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_BB_Squeeze_Breakout_4hEMA50_VolumeFilter"
timeframe = "1h"
leverage = 1.0