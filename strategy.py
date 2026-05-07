#!/usr/bin/env python3
name = "4h_1d_Keltner_Channel_Squeeze_Breakout"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtools import atr
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily Keltner Channel (20 EMA + 2*ATR)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # EMA(20) on daily close
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR(10) on daily
    atr_10_1d = atr(high_1d, low_1d, close_1d, 10)
    
    # Keltner bands
    upper_1d = ema_20_1d + 2 * atr_10_1d
    lower_1d = ema_20_1d - 2 * atr_10_1d
    
    # Align to 4h
    ema_20_1d_aligned = align_ltf_to_htf(prices, df_1d, ema_20_1d)
    upper_1d_aligned = align_ltf_to_htf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_ltf_to_htf(prices, df_1d, lower_1d)
    
    # 4h Bollinger Band width (20, 2) for squeeze detection
    close_series = pd.Series(close)
    sma_20 = close_series.rolling(window=20, min_periods=20).mean().values
    std_20 = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    bb_width = (bb_upper - bb_lower) / sma_20
    
    # Bollinger Band width percentile (50 lookback) for squeeze
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: np.percentile(x, 50) if len(x) == 50 else np.nan, raw=False
    ).values
    
    # Squeeze condition: BB width below 50th percentile
    squeeze = bb_width < bb_width_percentile
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Wait for BB width percentile and Bollinger
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_1d_aligned[i]) or np.isnan(upper_1d_aligned[i]) or 
            np.isnan(lower_1d_aligned[i]) or np.isnan(sma_20[i]) or np.isnan(std_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for breakout after squeeze
            if squeeze[i-1]:  # Was in squeeze
                # Breakout above upper Keltner with volume confirmation
                vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
                if not np.isnan(vol_ma_20[i]) and volume[i] > vol_ma_20[i] * 1.5:
                    if close[i] > upper_1d_aligned[i]:
                        signals[i] = 0.25
                        position = 1
                    elif close[i] < lower_1d_aligned[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Exit: return to middle (EMA) or opposite band
            if close[i] < ema_20_1d_aligned[i] or close[i] > upper_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: return to middle (EMA) or opposite band
            if close[i] > ema_20_1d_aligned[i] or close[i] < lower_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Bollinger Band squeeze + daily Keltner channel breakout
# - Bollinger Band squeeze indicates low volatility, priming for breakout
# - Breakout above/below daily Keltner channel (EMA20 ± 2*ATR) with volume
# - Works in both bull and bear markets by capturing volatility expansion
# - Daily timeframe provides stable volatility bands less prone to whipsaw
# - Position size 0.25 targets ~25-40 trades/year, minimizing fee drag
# - Exit when price returns to daily EMA (mean reversion) or breaks opposite band