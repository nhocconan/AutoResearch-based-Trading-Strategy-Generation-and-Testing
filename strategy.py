#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily trend with 1-week trend filter and volume confirmation.
# Uses EMA20/EMA50 on daily for trend alignment, weekly EMA20 for higher timeframe bias,
# and volume > 1.3x 20-day average to confirm momentum. Designed to capture
# sustained trends in both bull and bear markets while avoiding chop.
# Target: 20-40 trades per year on daily timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for trend and volume
    df_1d = get_htf_data(prices, '1d')
    
    # Daily EMA(20) for short-term trend
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Daily EMA(50) for long-term trend
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Weekly EMA(20) for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Daily volume and 20-day average
    volume = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma_20 == 0, 1, vol_ma_20)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_20_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_trend_short = ema_20_1d_aligned[i]
        ema_trend_long = ema_50_1d_aligned[i]
        ema_trend_weekly = ema_20_1w_aligned[i]
        atr = atr_14_1d_aligned[i]
        vol_ratio_1d = vol_ratio_aligned[i]
        
        # Multi-timeframe trend alignment
        # Uptrend: price above both daily EMAs and weekly EMA
        trend_up = (ema_trend_short > ema_trend_long) and (ema_trend_short > ema_trend_weekly)
        # Downtrend: price below both daily EMAs and weekly EMA
        trend_down = (ema_trend_short < ema_trend_long) and (ema_trend_short < ema_trend_weekly)
        
        # Volatility filter: avoid extremes
        atr_ma_20 = pd.Series(atr_14_1d_aligned).rolling(window=20, min_periods=20).mean().values[i]
        vol_filter = (atr > 0.5 * atr_ma_20) and (atr < 3.0 * atr_ma_20)
        
        # Volume confirmation
        vol_filter = vol_filter and (vol_ratio_1d > 1.3)
        
        if position == 0:
            # Enter long in strong uptrend with volume
            if trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short in strong downtrend with volume
            elif trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend breakdown or volatility spike
            if (not trend_up) or (atr > 3.5 * atr_ma_20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend breakdown or volatility spike
            if (not trend_down) or (atr > 3.5 * atr_ma_20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_EMA20_50_Trend_Volume_Filter_v1"
timeframe = "1d"
leverage = 1.0