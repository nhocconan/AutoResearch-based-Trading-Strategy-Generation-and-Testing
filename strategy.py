#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d trend filter and volume confirmation.
# In low volatility regimes (BB Width < 20th percentile), price is primed for breakout.
# We enter long when price breaks above upper BB with volume spike and 1d EMA34 uptrend,
# short when price breaks below lower BB with volume spike and 1d EMA34 downtrend.
# This captures explosive moves after consolidation, works in both bull and bear markets.

name = "6h_BollingerSqueeze_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Bollinger Bands (20, 2) on 6h
    bb_period = 20
    bb_std = 2
    sma_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma_20 + (bb_std * std_20)
    lower_bb = sma_20 - (bb_std * std_20)
    bb_width = ((upper_bb - lower_bb) / sma_20) * 100  # as percentage
    
    # Calculate BB Width percentile rank (20-period lookback)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=100, min_periods=100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50, raw=False
    ).values
    # Handle NaN values in percentile calculation
    bb_width_percentile = np.where(np.isnan(bb_width_percentile), 50, bb_width_percentile)
    
    # Volume spike: current volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if np.isnan(close[i]) or np.isnan(sma_20[i]) or np.isnan(std_20[i]) or \
           np.isnan(ema_34_aligned[i]) or np.isnan(bb_width_percentile[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Squeeze condition: BB Width below 20th percentile (low volatility)
        squeeze_active = bb_width_percentile[i] < 20
        
        # Breakout conditions
        long_breakout = close[i] > upper_bb[i]
        short_breakout = close[i] < lower_bb[i]
        
        # Trend filter from 1d EMA34
        ema_trend = ema_34_aligned[i]
        uptrend = close[i] > ema_trend
        downtrend = close[i] < ema_trend
        
        # Generate signals
        if position == 0:
            # Enter long: squeeze breakout up + uptrend + volume spike
            if squeeze_active and long_breakout and uptrend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: squeeze breakout down + downtrend + volume spike
            elif squeeze_active and short_breakout and downtrend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters Bollinger Bands or trend changes
            if close[i] < sma_20[i] or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters Bollinger Bands or trend changes
            if close[i] > sma_20[i] or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals