#!/usr/bin/env python3
# Hypothesis: 6h Bollinger Band squeeze breakout with 1d trend filter and volume confirmation.
# Long when price breaks above upper BB after low volatility (BBW < 20th percentile), close > 1d EMA50, volume > 1.5x 20-bar avg.
# Short when price breaks below lower BB after low volatility, close < 1d EMA50, volume > 1.5x 20-bar avg.
# Uses discrete sizing 0.25 to target 50-150 total trades over 4 years on 6h timeframe.
# Bollinger Band squeeze identifies low volatility primed for breakout; 1d EMA50 filters direction; volume confirms momentum.
# Designed for fewer, higher-quality trades to avoid fee drag while working in both bull and bear markets.

name = "6h_BBand_Squeeze_Breakout_1dEMA50_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Bollinger Bands (20, 2)
    lookback_bb = 20
    bb_mid = pd.Series(close).rolling(window=lookback_bb, min_periods=lookback_bb).mean().values
    bb_std = pd.Series(close).rolling(window=lookback_bb, min_periods=lookback_bb).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band Width percentile (200 lookback for regime)
    lookback_percentile = 200
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=lookback_percentile, min_periods=lookback_percentile//2).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) == lookback_percentile else np.nan, raw=False
    ).values
    bb_width_percentile = np.where(np.isnan(bb_width_percentile), 50, bb_width_percentile)  # fill NaN with median
    
    # Average volume for confirmation (20-period)
    lookback_vol = 20
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback_bb, lookback_percentile, lookback_vol) + 1
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_width_percentile[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper BB after squeeze (BBW < 20th percentile), close > 1d EMA50, volume spike
            if (close[i] > bb_upper[i] and 
                bb_width_percentile[i] < 20 and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower BB after squeeze, close < 1d EMA50, volume spike
            elif (close[i] < bb_lower[i] and 
                  bb_width_percentile[i] < 20 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below lower BB OR volume drops below average
            if (close[i] < bb_lower[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above upper BB OR volume drops below average
            if (close[i] > bb_upper[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals