#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud + 1d ADX Trend Filter
# Ichimoku: TK Cross + Cloud Filter (price above/below cloud)
# In trending markets (ADX > 25): trade with Kumo breakout direction
# In ranging markets (ADX < 20): fade at cloud edges (Kumo twist/rejection)
# Uses standard Ichimoku parameters (9,26,52) on 6H chart
# Target: 60-150 trades over 4 years with trend/range adaptation

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku components (standard parameters)
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 52 periods
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): close shifted -22 periods (not used for signals)
    
    # Calculate 1d ADX (14-period) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for ADX
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr2_1d[0] = 0
    tr3_1d[0] = 0
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    
    # Directional Movement
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    # Pre-calculate cloud boundaries for current price (no look-ahead)
    # Senkou Span A/B shifted forward by 26 periods
    senkou_a_shifted = np.roll(senkou_span_a, 26)
    senkou_b_shifted = np.roll(senkou_span_b, 26)
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_shifted, senkou_b_shifted)
    cloud_bottom = np.minimum(senkou_a_shifted, senkou_b_shifted)
    
    for i in range(52, n):  # Start after Senkou Span B calculation
        # Get aligned 1d ADX
        adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)[i]
        
        # Check for NaN values
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(adx_1d_aligned)):
            continue
        
        # Ichimoku signals
        tk_cross = tenkan_sen[i] - kijun_sen[i]  # Positive = bullish cross
        price_above_cloud = close[i] > cloud_top[i]
        price_below_cloud = close[i] < cloud_bottom[i]
        price_in_cloud = (close[i] >= cloud_bottom[i]) & (close[i] <= cloud_top[i])
        
        # Kumo twist (Senkou A/B cross) - potential trend change
        kumo_twist = senkou_a_shifted[i] - senkou_b_shifted[i]
        kumo_twist_prev = senkou_a_shifted[i-1] - senkou_b_shifted[i-1] if i > 0 else 0
        kumo_twist_change = kumo_twist * kumo_twist_prev < 0 and abs(kumo_twist) > 0.001
        
        # Regime classification
        ranging = adx_1d_aligned < 20
        trending = adx_1d_aligned > 25
        
        if position == 0:  # No position - look for entries
            if trending:
                # In trend: trade Kumo breakouts
                if tk_cross > 0 and price_above_cloud:  # Bullish TK cross + above cloud
                    position = 1
                    signals[i] = position_size
                elif tk_cross < 0 and price_below_cloud:  # Bearish TK cross + below cloud
                    position = -1
                    signals[i] = -position_size
            elif ranging:
                # In range: fade at cloud edges, avoid Kumo twist
                if not kumo_twist_change:  # Avoid trading during Kumo twist
                    if price_above_cloud and tk_cross < 0:  # At cloud top, bearish TK cross
                        position = -1
                        signals[i] = -position_size
                    elif price_below_cloud and tk_cross > 0:  # At cloud bottom, bullish TK cross
                        position = 1
                        signals[i] = position_size
        elif position == 1:  # Long position - exit conditions
            # Exit if price falls below cloud or TK cross turns bearish
            if price_below_cloud or tk_cross < 0:
                position = 0
                signals[i] = 0.0
        elif position == -1:  # Short position - exit conditions
            # Exit if price rises above cloud or TK cross turns bullish
            if price_above_cloud or tk_cross > 0:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "6h_Ichimoku_1dADX_CloudFilter"
timeframe = "6h"
leverage = 1.0