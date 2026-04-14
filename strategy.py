#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Keltner Channel breakout with volume confirmation and ADX trend filter.
# Long when price breaks above upper Keltner Channel on 1d timeframe, ADX > 25 (trending), and volume > 1.5x average.
# Short when price breaks below lower Keltner Channel on 1d timeframe, ADX > 25, and volume > 1.5x average.
# Exit when price returns to Keltner middle line (EMA) or ADX drops below 20 (trend weakening).
# Uses Keltner Channels for volatility-based breakout signals, ADX for trend strength confirmation,
# and volume for institutional participation confirmation. Designed to work in both bull and bear markets
# by only trading in trending conditions (ADX > 25) and avoiding choppy markets.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Keltner Channel and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough for EMA(20), ATR(10), and ADX(14)
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA(20) for middle line
    ema_20 = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate ATR(10)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner Channel (20, 2)
    kc_middle = ema_20
    kc_upper = ema_20 + 2 * atr_10
    kc_lower = ema_20 - 2 * atr_10
    
    # Calculate ADX (14)
    # True Range
    tr1_adx = high_1d[1:] - low_1d[1:]
    tr2_adx = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_adx = np.abs(low_1d[1:] - close_1d[:-1])
    tr_adx = np.concatenate([[np.nan], np.maximum(tr1_adx, np.maximum(tr2_adx, tr3_adx))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr_adx).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align indicators to 12h timeframe
    kc_upper_aligned = align_htf_to_ltf(prices, df_1d, kc_upper)
    kc_lower_aligned = align_htf_to_ltf(prices, df_1d, kc_lower)
    kc_middle_aligned = align_htf_to_ltf(prices, df_1d, kc_middle)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: 1.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(34, 20)  # Need ADX and EMA periods
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(kc_upper_aligned[i]) or 
            np.isnan(kc_lower_aligned[i]) or
            np.isnan(kc_middle_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Weak trend filter: ADX < 20 indicates trend weakening
        weak_trend = adx_aligned[i] < 20
        
        if position == 0:
            # Look for Keltner Channel breakouts in strong trend
            # Long: price breaks above upper KC AND strong trend AND volume confirmation
            if (close[i] > kc_upper_aligned[i] and 
                strong_trend and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower KC AND strong trend AND volume confirmation
            elif (close[i] < kc_lower_aligned[i] and 
                  strong_trend and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle KC or trend weakens
            if (close[i] <= kc_middle_aligned[i] or 
                weak_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to middle KC or trend weakens
            if (close[i] >= kc_middle_aligned[i] or 
                weak_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Keltner_Channel_ADX_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0