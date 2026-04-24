#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud + 1d ADX Regime Filter + Volume Confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d ADX(14) for regime filter (ADX > 25 = trending, ADX < 20 = ranging).
- Entry: Long when Tenkan-sen > Kijun-sen AND price > Senkou Span A/B (cloud) AND ADX > 25 AND volume > 1.5 * 6h volume MA(20);
         Short when Tenkan-sen < Kijun-sen AND price < Senkou Span A/B (cloud) AND ADX > 25 AND volume > 1.5 * 6h volume MA(20).
- Exit: Long exits when Tenkan-sen < Kijun-sen OR price < cloud; Short exits when Tenkan-sen > Kijun-sen OR price > cloud.
- Signal size: 0.25 discrete to balance capture and fee control.
- Ichimoku provides trend, momentum, and support/resistance; ADX filters strong trends; volume confirms conviction.
- Works in bull (buying above cloud in uptrend) and bear (selling below cloud in downtrend) with reduced whipsaws in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+,
    period = 14
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Ichimoku components on 6h data
    # Conversion Line (Tenkan-sen): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Base Line (Kijun-sen): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Leading Span A (Senkou Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Leading Span B (Senkou Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # For cloud, we use the current values of Senkou Span A/B (already shifted in calculation)
    # The cloud ahead is senkou_span_a and senkou_span_b, but for price vs cloud we compare to current cloud
    # Actually, the cloud is plotted 26 periods ahead, so we need to compare current price to cloud that was plotted 26 periods ago
    # But for simplicity in live trading, we use current Senkou Span A/B as cloud boundaries
    # We'll consider price above cloud if price > max(senkou_span_a, senkou_span_b)
    # and price below cloud if price < min(senkou_span_a, senkou_span_b)
    
    # Volume MA(20) for 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(26, 20)  # Ichimoku needs 26, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Ichimoku signals
        tenkan_above_kijun = tenkan_sen[i] > kijun_sen[i]
        tenkan_below_kijun = tenkan_sen[i] < kijun_sen[i]
        
        # Cloud boundaries
        upper_cloud = np.maximum(senkou_span_a[i], senkou_span_b[i])
        lower_cloud = np.minimum(senkou_span_a[i], senkou_span_b[i])
        
        price_above_cloud = curr_close > upper_cloud
        price_below_cloud = curr_close < lower_cloud
        
        # Regime filter from 1d ADX
        trending = adx_aligned[i] > 25
        ranging = adx_aligned[i] < 20  # We'll use this to avoid trading in ranging markets
        
        # Volume confirmation: 1.5x threshold
        vol_confirm = curr_volume > 1.5 * vol_ma[i]
        
        if position == 0:
            # Check for entry signals - only trade in trending markets (ADX > 25)
            if trending and vol_confirm:
                # Long: bullish TK cross AND price above cloud
                if tenkan_above_kijun and price_above_cloud:
                    signals[i] = 0.25
                    position = 1
                # Short: bearish TK cross AND price below cloud
                elif tenkan_below_kijun and price_below_cloud:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when bearish TK cross OR price below cloud
            if tenkan_below_kijun or price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when bullish TK cross OR price above cloud
            if tenkan_above_kijun or price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_1dADX_Regime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0