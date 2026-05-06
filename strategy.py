#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Keltner Channel breakout with 12-hour ATR volatility filter and volume confirmation
# Uses Keltner Channel (EMA20 + 2*ATR) for volatility-based breakout detection
# Requires breakout with close outside channel + volume > 1.5x 20-bar average
# Uses 12-hour ATR ratio (current ATR/20-period average) to filter for expanding volatility regimes
# Designed for 6h timeframe to target 50-150 total trades over 4 years (12-37/year)
# Works in both bull/bear: captures volatility expansion breakouts, avoids low-volatility chop

name = "6h_KeltnerBreakout_12hATRRatio_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12-period ATR for 12h timeframe
    tr1_12h = np.abs(high_12h[1:] - low_12h[1:])
    tr2_12h = np.abs(high_12h[1:] - close_12h[:-1])
    tr3_12h = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.concatenate([[np.nan], np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))])
    atr_12h = pd.Series(tr_12h).rolling(window=12, min_periods=12).mean().values
    
    # Calculate 20-period average ATR for volatility regime filter
    atr_ma_20 = pd.Series(atr_12h).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_12h / atr_ma_20  # Current ATR relative to 20-period average
    
    # Calculate Keltner Channel on 6h timeframe
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate ATR for Keltner Channel (using same period as EMA for consistency)
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_kc = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Keltner Channel bounds
    kc_upper = ema_20 + (2.0 * atr_kc)
    kc_lower = ema_20 - (2.0 * atr_kc)
    
    # Calculate volume confirmation filter (>1.5x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Align HTF indicators to 6h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_12h, atr_ratio)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(ema_20[i]) or np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or 
            np.isnan(atr_ratio_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: close above upper Keltner band AND expanding volatility (ATR ratio > 1.2) AND volume confirmation
            if (close[i] > kc_upper[i] and atr_ratio_aligned[i] > 1.2 and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: close below lower Keltner band AND expanding volatility (ATR ratio > 1.2) AND volume confirmation
            elif (close[i] < kc_lower[i] and atr_ratio_aligned[i] > 1.2 and volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: close below EMA20 (mean reversion to the mean)
            if close[i] < ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: close above EMA20 (mean reversion to the mean)
            if close[i] > ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals