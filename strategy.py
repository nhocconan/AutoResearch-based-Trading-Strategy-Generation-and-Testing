#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy combining daily Camarilla pivot levels with volume spike confirmation and
# 1-week ADX trend filter. Long when price breaks above daily Camarilla H3 level with ADX>25 and
# volume > 1.5x 20-period average. Short when price breaks below daily Camarilla L3 level with
# ADX>25 and volume confirmation. Exit when price returns to daily close or ADX drops below 20.
# Uses Camarilla levels for intraday support/resistance, ADX for trend strength, and volume for
# confirmation. Designed for 12h timeframe to capture multi-day moves while minimizing trades.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to stay well under fee drag limits.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for Camarilla levels and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Daily Camarilla Pivot Levels
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: H3/L3 are key breakout levels
    # H3 = Close + 1.1 * (High - Low) / 2
    # L3 = Close - 1.1 * (High - Low) / 2
    camarilla_h3 = close_1d + 1.1 * range_1d / 2.0
    camarilla_l3 = close_1d - 1.1 * range_1d / 2.0
    
    # Daily ADX for trend strength filter
    # Calculate True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Calculate Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    atr_period = 14
    tr_smooth = pd.Series(tr).ewm(alpha=1/atr_period, adjust=False).values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/atr_period, adjust=False).values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/atr_period, adjust=False).values
    
    # Calculate DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # Calculate DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/atr_period, adjust=False).values
    
    # Load weekly data ONCE for additional trend filter (optional)
    # Using daily ADX as primary trend filter
    
    # Align indicators to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 20)  # Need ADX and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(close_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        weak_trend = adx_aligned[i] < 20  # Exit when trend weakens
        
        if position == 0:
            # Look for Camarilla breakouts
            # Long: price breaks above H3 with strong trend and volume
            if (close[i] > camarilla_h3_aligned[i] and 
                strong_trend and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below L3 with strong trend and volume
            elif (close[i] < camarilla_l3_aligned[i] and 
                  strong_trend and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to daily close or trend weakens
            if (close[i] <= close_1d_aligned[i] or 
                weak_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to daily close or trend weakens
            if (close[i] >= close_1d_aligned[i] or 
                weak_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Camarilla_H3L3_ADX_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0