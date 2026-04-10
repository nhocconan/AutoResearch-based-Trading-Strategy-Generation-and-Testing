#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud + 1d ADX trend filter
# - Primary signal: Tenkan-sen/Kijun-sen cross on 6h with price above/below Kumo (cloud)
# - Trend filter: 1d ADX > 25 ensures we only trade in strong trends (avoids whipsaws in ranging markets)
# - Volume filter: 6h volume > 1.5x 20-period average (confirms momentum)
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 2.0x ATR(14) on 6h
# - Target: 12-37 trades/year (50-150 total over 4 years)
# - Works in bull/bear: Ichimoku captures trend direction and support/resistance; ADX filter ensures we only trade when trends are strong

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
    
    # Pre-compute 1d ADX(14) for trend strength filter
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
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx[di_plus + di_minus == 0] = 0
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_strong = adx > 25  # Strong trend
    
    # Align ADX to 6h timeframe
    adx_strong_aligned = align_htf_to_ltf(prices, df_1d, adx_strong)
    
    # Pre-compute 6h Ichimoku components
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_6h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_6h).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Current Kumo (Cloud) boundaries: Senkou Span A and B shifted back 26 periods
    senkou_a_lagged = np.roll(senkou_a, 26)
    senkou_b_lagged = np.roll(senkou_b, 26)
    senkou_a_lagged[:26] = np.nan
    senkou_b_lagged[:26] = np.nan
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_lagged, senkou_b_lagged)
    cloud_bottom = np.minimum(senkou_a_lagged, senkou_b_lagged)
    
    # Pre-compute 6h volume spike filter
    volume_6h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_6h > (1.5 * avg_volume_20)
    
    # Pre-compute 6h ATR(14) for stoploss
    tr1_6h = high_6h - low_6h
    tr2_6h = np.abs(high_6h - np.roll(close_6h, 1))
    tr3_6h = np.abs(low_6h - np.roll(close_6h, 1))
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    tr_6h[0] = tr1_6h[0]
    atr_14 = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(adx_strong_aligned[i]) or np.isnan(vol_spike[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Tenkan-sen crosses below Kijun-sen OR price falls below cloud OR stoploss hit
            if (tenkan_sen[i] < kijun_sen[i] or close_6h[i] < cloud_bottom[i] or 
                close_6h[i] < entry_price - 2.0 * atr_14[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Tenkan-sen crosses above Kijun-sen OR price rises above cloud OR stoploss hit
            if (tenkan_sen[i] > kijun_sen[i] or close_6h[i] > cloud_top[i] or 
                close_6h[i] > entry_price + 2.0 * atr_14[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Ichimoku signals with trend and volume filters
            if vol_spike[i] and adx_strong_aligned[i]:
                # Long: Tenkan-sen crosses above Kijun-sen AND price above cloud
                if (tenkan_sen[i] > kijun_sen[i] and tenkan_sen[i-1] <= kijun_sen[i-1] and 
                    close_6h[i] > cloud_top[i]):
                    position = 1
                    entry_price = close_6h[i]
                    signals[i] = 0.25
                # Short: Tenkan-sen crosses below Kijun-sen AND price below cloud
                elif (tenkan_sen[i] < kijun_sen[i] and tenkan_sen[i-1] >= kijun_sen[i-1] and 
                      close_6h[i] < cloud_bottom[i]):
                    position = -1
                    entry_price = close_6h[i]
                    signals[i] = -0.25
    
    return signals