#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation
# - Uses Ichimoku (Tenkan, Kijun, Senkou Span A/B) on 6h for trend and momentum
# - 1d ADX > 25 to ensure we trade with higher timeframe trend (avoid chop)
# - Volume confirmation: current 6h volume > 1.5x 20-period average
# - Long when price > Cloud AND Tenkan > Kijun (bullish alignment)
# - Short when price < Cloud AND Tenkan < Kijun (bearish alignment)
# - Exit when price crosses Tenkan-Kijun line (TK cross) or volume drops
# - Designed for 6h timeframe: targets 12-37 trades/year (50-150 over 4 years)
# - Works in bull/bear markets: 1d ADX filter ensures higher timeframe trend alignment
# - Uses discrete position sizing (0.25) to minimize fee churn

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
    
    # Pre-compute 1d ADX(14) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute 6h Ichimoku components
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_6h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_6h).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # The Cloud: between Senkou Span A and B
    # For plotting, Senkou spans are shifted 26 periods ahead
    # For trading, we use current Senkou values (already represent future cloud)
    
    # Pre-compute 6h volume confirmation
    volume_6h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_6h > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Need 52 periods for Senkou B
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou A and B)
        upper_cloud = max(senkou_a[i], senkou_b[i])
        lower_cloud = min(senkou_a[i], senkou_b[i])
        
        if position == 1:  # Long position
            # Exit: price falls below Cloud OR TK cross turns bearish OR volume drops
            if (close_6h[i] < lower_cloud or 
                tenkan[i] < kijun[i] or 
                not vol_spike[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above Cloud OR TK cross turns bullish OR volume drops
            if (close_6h[i] > upper_cloud or 
                tenkan[i] > kijun[i] or 
                not vol_spike[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Ichimoku signal with trend and volume filters
            if vol_spike[i] and adx_aligned[i] > 25:
                # Bullish: price above Cloud AND Tenkan > Kijun
                if close_6h[i] > upper_cloud and tenkan[i] > kijun[i]:
                    position = 1
                    signals[i] = 0.25
                # Bearish: price below Cloud AND Tenkan < Kijun
                elif close_6h[i] < lower_cloud and tenkan[i] < kijun[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals