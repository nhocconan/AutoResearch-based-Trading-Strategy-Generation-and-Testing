#!/usr/bin/env python3
# 6h_adaptive_keltner_breakout_1d_trend_volume_v1
# Hypothesis: On 6h timeframe, adaptive Keltner breakout with volume confirmation and 1d EMA50 trend alignment captures momentum moves. 
# The adaptive ATR multiplier (based on volatility regime) prevents whipsaws in ranging markets while allowing breakouts in trending periods.
# Works in both bull and bear markets by following the higher timeframe trend.
# Entry: Long when price > upper Keltner band + volume > 1.5x 20-period average + price > 1d EMA50
# Entry: Short when price < lower Keltner band + volume > 1.5x 20-period average + price < 1d EMA50
# Exit: Price crosses back below/above middle line (EMA20) or trend reversal
# Position sizing: 0.25 long, -0.25 short

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_adaptive_keltner_breakout_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # EMA20 for middle line (60 periods for 6m data)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR for Keltner bands (20 periods)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volatility regime detection: ATR ratio (short/long)
    atr_short = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr_long = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    atr_ratio = atr_short / (atr_long + 1e-10)
    
    # Adaptive multiplier: higher in low volatility (tighter bands), lower in high volatility (wider bands)
    # Normal range 0.5-2.0, inverted so low vol = higher multiplier
    multiplier = np.clip(2.0 - (atr_ratio - 0.5) * 2, 0.5, 2.5)
    
    # Keltner bands
    upper_band = ema_20 + multiplier * atr
    lower_band = ema_20 - multiplier * atr
    
    # Volume filter: 6h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema_20[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price below EMA20 OR price below 1d EMA50 (trend reversal)
            if (close[i] < ema_20[i]) or (close[i] < ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: price above EMA20 OR price above 1d EMA50 (trend reversal)
            if (close[i] > ema_20[i]) or (close[i] > ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: price > upper Keltner band + volume + price > 1d EMA50
            if (close[i] > upper_band[i]) and volume_filter[i] and (close[i] > ema_50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price < lower Keltner band + volume + price < 1d EMA50
            elif (close[i] < lower_band[i]) and volume_filter[i] and (close[i] < ema_50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals