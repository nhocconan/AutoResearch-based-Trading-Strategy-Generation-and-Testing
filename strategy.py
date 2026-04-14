#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1-week Keltner Channel breakout with volume confirmation and ADX trend filter.
# Long when price closes above upper Keltner Channel on weekly timeframe, ADX > 25 (trending), and volume > 1.5x average.
# Short when price closes below lower Keltner Channel on weekly timeframe, ADX > 25, and volume > 1.5x average.
# Exit when price returns to Keltner Channel middle line or ADX drops below 20 (trend weakening).
# Uses Keltner Channels (ATR-based) for volatility-based breakout signals, ADX for trend strength confirmation,
# and volume for institutional participation confirmation. Designed to work in both bull and bear markets
# by only trading in trending conditions (ADX > 25) and avoiding choppy markets.
# Target: 10-25 trades/year per symbol (40-100 total over 4 years) to minimize fee drain.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE for Keltner Channels and ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need enough for KC(20,2) and ADX(14)
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Keltner Channels (20, 2)
    kc_middle = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    atr = pd.Series(high_1w - low_1w).rolling(window=20, min_periods=20).mean().values
    kc_upper = kc_middle + 2 * atr
    kc_lower = kc_middle - 2 * atr
    
    # Calculate ADX (14)
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align indicators to 1d timeframe
    kc_upper_aligned = align_htf_to_ltf(prices, df_1w, kc_upper)
    kc_lower_aligned = align_htf_to_ltf(prices, df_1w, kc_lower)
    kc_middle_aligned = align_htf_to_ltf(prices, df_1w, kc_middle)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume confirmation: 1.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(34, 20)  # Need ADX and KC periods
    
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
            # Long: price closes above upper KC AND strong trend AND volume confirmation
            if (close[i] > kc_upper_aligned[i] and 
                strong_trend and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price closes below lower KC AND strong trend AND volume confirmation
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

name = "1d_1w_Keltner_Channel_ADX_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0