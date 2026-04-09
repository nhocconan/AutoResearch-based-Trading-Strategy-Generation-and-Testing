#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud + 1d ADX trend filter + volume confirmation
# - Primary signal: Ichimoku Tenkan/Kijun cross (TK cross) on 6h timeframe
# - Trend filter: 1d ADX > 25 ensures we only trade in trending markets (avoids whipsaws in ranges)
# - Cloud filter: Price must be above/below the Kumo (cloud) for confirmation
# - Volume confirmation: 6h volume > 20-period median volume (avoid low-participation signals)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: Ichimoku captures trends, ADX filter avoids false signals in low volatility, cloud acts as dynamic support/resistance

name = "6h_1d_ichimoku_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d ADX for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Movement
    dm_plus = pd.Series(np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                                 np.maximum(high_1d - np.roll(high_1d, 1), 0), 0))
    dm_minus = pd.Series(np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                                  np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0))
    
    # Smoothed DM
    dm_plus_smooth = dm_plus.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    dm_minus_smooth = dm_minus.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx_values = adx.values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Pre-compute Ichimoku components on 6h timeframe
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_6h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_6h).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Current Kumo (cloud) boundaries: Senkou Span A and B from 26 periods ago
    senkou_a_lag = np.roll(senkou_a, 26)
    senkou_b_lag = np.roll(senkou_b, 26)
    # For first 26 periods, use available values (will be NaN until enough data)
    senkou_a_lag[:26] = senkou_a[:26]
    senkou_b_lag[:26] = senkou_b[:26]
    
    # Kumo top and bottom
    kumo_top = np.maximum(senkou_a_lag, senkou_b_lag)
    kumo_bottom = np.minimum(senkou_a_lag, senkou_b_lag)
    
    # 6h volume regime: volume > 20-period median volume
    volume = prices['volume'].values
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > median_volume_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or
            np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: TK cross down (tenkan < kijun) OR price falls below kumo bottom OR ADX < 20 (trend weakening)
            if (tenkan[i] < kijun[i] or 
                close_6h[i] < kumo_bottom[i] or 
                adx_aligned[i] < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TK cross up (tenkan > kijun) OR price rises above kumo top OR ADX < 20 (trend weakening)
            if (tenkan[i] > kijun[i] or 
                close_6h[i] > kumo_top[i] or 
                adx_aligned[i] < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for TK cross with volume confirmation, ADX filter, and cloud filter
            # Bullish TK cross: tenkan crosses above kijun
            tk_bullish = (tenkan[i] > kijun[i]) and (tenkan[i-1] <= kijun[i-1])
            # Bearish TK cross: tenkan crosses below kijun
            tk_bearish = (tenkan[i] < kijun[i]) and (tenkan[i-1] >= kijun[i-1])
            
            # Long conditions: bullish TK cross AND price above kumo (bullish cloud) AND ADX > 25 AND volume regime
            if (tk_bullish and 
                close_6h[i] > kumo_top[i] and 
                adx_aligned[i] > 25 and 
                volume_regime[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: bearish TK cross AND price below kumo (bearish cloud) AND ADX > 25 AND volume regime
            elif (tk_bearish and 
                  close_6h[i] < kumo_bottom[i] and 
                  adx_aligned[i] > 25 and 
                  volume_regime[i]):
                position = -1
                signals[i] = -0.25
    
    return signals