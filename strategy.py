#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band squeeze breakout with volume confirmation and ADX trend filter
# Bollinger Band width < 20th percentile indicates low volatility squeeze (mean reversion setup)
# Breakout above upper band or below lower band with volume > 1.5x average triggers entry
# ADX > 25 confirms trend strength to avoid false breakouts in chop
# Works in bull markets (breakouts continue trends) and bear markets (breakouts catch reversals)
# Designed for low trade frequency (target 15-25/year) with high win rate

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d Bollinger Bands (20, 2)
    sma20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    bb_width = (upper - lower) / sma20  # Normalized width
    
    # 1d Bollinger Band width percentile (20-period lookback)
    bb_width_percentile = pd.Series(bb_width).rolling(window=20, min_periods=20).rank(pct=True).values
    
    # 1d average volume (20-period)
    avg_volume = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # 1d ADX(14) for trend strength
    tr1 = np.maximum(high_1d[1:], low_1d[:-1]) - np.minimum(high_1d[1:], low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr_1d + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr_1d + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align indicators to 12h timeframe
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    avg_volume_aligned = align_htf_to_ltf(prices, df_1d, avg_volume)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_width_percentile_aligned[i]) or np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or np.isnan(avg_volume_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            continue
        
        # Squeeze condition: BB width in lowest 20% (tight bands)
        squeeze = bb_width_percentile_aligned[i] <= 0.2
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume[i] > 1.5 * avg_volume_aligned[i]
        
        # Trend filter: ADX > 25
        strong_trend = adx_aligned[i] > 25
        
        # Breakout conditions
        if squeeze and volume_confirm and strong_trend:
            # Long breakout above upper band
            if close[i] > upper_aligned[i] and position <= 0:
                position = 1
                signals[i] = position_size
            # Short breakout below lower band
            elif close[i] < lower_aligned[i] and position >= 0:
                position = -1
                signals[i] = -position_size
        
        # Exit conditions: reverse signal or loss of momentum
        elif position == 1 and close[i] < sma20_aligned[i]:  # Return to mean
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > sma20_aligned[i]:  # Return to mean
            position = 0
            signals[i] = 0.0
    
    return signals

# Align SMA20 for exit condition
sma20 = pd.Series(df_1d['close'].values).rolling(window=20, min_periods=20).mean().values
sma20_aligned = align_htf_to_ltf(prices, df_1d, sma20)

name = "12h_BollingerSqueeze_Breakout_Volume_ADX"
timeframe = "12h"
leverage = 1.0