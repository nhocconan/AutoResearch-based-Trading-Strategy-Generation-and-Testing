#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily Williams Alligator for trend direction + daily ATR breakout for entry timing + volume confirmation
# Long when price > Alligator Jaw (blue line) AND price breaks above ATR(14) upper band AND volume > 1.5 * avg_volume(20)
# Short when price < Alligator Jaw (blue line) AND price breaks below ATR(14) lower band AND volume > 1.5 * avg_volume(20)
# Exit when price crosses back below/above Alligator Jaw OR ATR ratio < 1.0 (low volatility)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Williams Alligator (SMAs smoothed) provides robust trend filter that whipsaws less in ranging markets
# ATR breakout captures momentum expansion after consolidation
# Volume confirmation ensures breakout authenticity
# Works in bull markets (trend + breakout) and bear markets (trend + breakdown)

name = "12h_WilliamsAlligator_ATR_Breakout_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data ONCE before loop for Williams Alligator and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for Alligator and ATR
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator (using SMAs with smoothing)
    # Jaw (blue line): 13-period SMA smoothed by 8 periods
    close_1d_series = pd.Series(close_1d)
    sma13 = close_1d_series.rolling(window=13, min_periods=13).mean()
    jaw = sma13.rolling(window=8, min_periods=8).mean().values  # Smoothed SMA
    
    # Calculate ATR(14) for volatility bands
    tr1 = high_1d[1:] - low_1d[:-1]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR bands (upper/lower) - using close as base
    atr_upper = close_1d + (atr * 1.0)  # 1*ATR above close
    atr_lower = close_1d - (atr * 1.0)  # 1*ATR below close
    
    # Align daily indicators to 12h timeframe (wait for completed daily bar)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    atr_upper_aligned = align_htf_to_ltf(prices, df_1d, atr_upper)
    atr_lower_aligned = align_htf_to_ltf(prices, df_1d, atr_lower)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 12h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(jaw_aligned[i]) or np.isnan(atr_upper_aligned[i]) or 
            np.isnan(atr_lower_aligned[i]) or np.isnan(avg_volume_20[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price > Alligator Jaw AND breaks above ATR upper band AND volume confirmation, in session
            if close[i] > jaw_aligned[i] and close[i] > atr_upper_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price < Alligator Jaw AND breaks below ATR lower band AND volume confirmation, in session
            elif close[i] < jaw_aligned[i] and close[i] < atr_lower_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below Alligator Jaw OR ATR ratio < 1.0 (low volatility)
            if close[i] < jaw_aligned[i] or (atr[i] / np.mean(atr[max(0, i-19):i+1]) < 1.0 if i >= 20 else False):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above Alligator Jaw OR ATR ratio < 1.0 (low volatility)
            if close[i] > jaw_aligned[i] or (atr[i] / np.mean(atr[max(0, i-19):i+1]) < 1.0 if i >= 20 else False):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals