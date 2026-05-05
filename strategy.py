#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using weekly ATR-based volatility breakout with 1d EMA50 trend filter and volume spike confirmation
# Long when price breaks above weekly ATR(14) upper band AND price > 1d EMA50 AND volume > 1.5 * avg_volume(20) on 12h
# Short when price breaks below weekly ATR(14) lower band AND price < 1d EMA50 AND volume > 1.5 * avg_volume(20) on 12h
# Exit when price crosses back below/above weekly ATR middle band OR volume drops below average
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Weekly ATR provides volatility-adjusted breakout levels that adapt to market conditions
# 1d EMA50 filters primary trend to avoid counter-trend trades
# Volume confirmation reduces false signals
# Works in bull markets (breakouts with uptrend) and bear markets (breakdowns with downtrend)

name = "12h_ATR_Volatility_Breakout_1dEMA50_VolumeSpike"
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
    
    # Get weekly data ONCE before loop for ATR calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:  # Need enough for ATR14
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly ATR(14)
    tr1 = np.maximum(high_1w[1:] - low_1w[1:], np.abs(high_1w[1:] - close_1w[:-1]))
    tr2 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.max([high_1w[0] - low_1w[0], np.abs(high_1w[0] - close_1w[0]), np.abs(low_1w[0] - close_1w[0])])], np.maximum(tr1, tr2)])
    atr_1w = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate weekly ATR bands: middle = SMA(20), upper = middle + 2*ATR, lower = middle - 2*ATR
    sma_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    atr_upper = sma_20_1w + (2.0 * atr_1w)
    atr_lower = sma_20_1w - (2.0 * atr_1w)
    atr_middle = sma_20_1w
    
    # Align weekly ATR bands to 12h timeframe (wait for completed weekly bar)
    atr_upper_aligned = align_htf_to_ltf(prices, df_1w, atr_upper)
    atr_lower_aligned = align_htf_to_ltf(prices, df_1w, atr_lower)
    atr_middle_aligned = align_htf_to_ltf(prices, df_1w, atr_middle)
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA50
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
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
        if (np.isnan(atr_upper_aligned[i]) or np.isnan(atr_lower_aligned[i]) or 
            np.isnan(atr_middle_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly ATR upper band, above 1d EMA50, volume confirmation, in session
            if close[i] > atr_upper_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly ATR lower band, below 1d EMA50, volume confirmation, in session
            elif close[i] < atr_lower_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below weekly ATR middle band OR volume drops below average
            if close[i] < atr_middle_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above weekly ATR middle band OR volume drops below average
            if close[i] > atr_middle_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals