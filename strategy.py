# 12h_1w_engulfing_pattern_volume_filter_v1
# Hypothesis: On 12h timeframe, weekly bullish/bearish engulfing patterns combined with volume confirmation (>1.5x average volume) and ADX filter (>25) capture high-probability trend continuations.
# Weekly engulfing patterns identify strong weekly reversals/continuations. Volume confirms institutional interest. ADX>25 ensures trades only in strong trends, reducing whipsaw.
# Target: 15-25 trades per year to minimize fee drag while capturing strong weekly moves.
# Works in bull markets (captures continuations) and bear markets (captures trend continuations during rallies/sell-offs).

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_engulfing_pattern_volume_filter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly OHLC for engulfing pattern detection
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    open_1w = df_1w['open'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly bullish and bearish engulfing patterns
    # Bullish engulfing: current week closes above prior week's open AND opens below prior week's close
    bullish_engulf = (close_1w > open_1w) & (open_1w < close_1w) & \
                     (close_1w > open_1w[:-1]) & (open_1w < close_1w[:-1]) & \
                     (close_1w > close_1w[:-1]) & (open_1w < open_1w[:-1])
    # Bearish engulfing: current week closes below prior week's open AND opens above prior week's close
    bearish_engulf = (close_1w < open_1w) & (open_1w > close_1w) & \
                     (close_1w < open_1w[:-1]) & (open_1w > close_1w[:-1]) & \
                     (close_1w < close_1w[:-1]) & (open_1w > open_1w[:-1])
    
    # Handle first week (no prior week)
    bullish_engulf[0] = False
    bearish_engulf[0] = False
    
    # Align weekly engulfing signals to 12h timeframe (use signal from prior week's close)
    bullish_engulf_aligned = align_htf_to_ltf(prices, df_1w, bullish_engulf.astype(float))
    bearish_engulf_aligned = align_htf_to_ltf(prices, df_1w, bearish_engulf.astype(float))
    
    # 12h volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 12h ADX for trend strength (14 period)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr_dm = np.concatenate([[np.nan], np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))])
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / pd.Series(tr_dm).rolling(window=14, min_periods=14).mean().values
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / pd.Series(tr_dm).rolling(window=14, min_periods=14).mean().values
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(bullish_engulf_aligned[i]) or np.isnan(bearish_engulf_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation (1.5x average)
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Trend filter: ADX > 25 (strong trend filter to reduce trades)
        trend_filter = adx[i] > 25
        
        # Long conditions: weekly bullish engulfing with volume and trend
        long_signal = volume_confirmed and trend_filter and bullish_engulf_aligned[i] > 0.5
        
        # Short conditions: weekly bearish engulfing with volume and trend
        short_signal = volume_confirmed and trend_filter and bearish_engulf_aligned[i] > 0.5
        
        # Exit when opposite engulfing pattern appears (pattern completion)
        exit_long = position == 1 and bearish_engulf_aligned[i] > 0.5
        exit_short = position == -1 and bullish_engulf_aligned[i] > 0.5
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals