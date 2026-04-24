#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1d EMA trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d EMA(34) for trend filter (defines bull/bear regime above/below price).
- Entry: Long when Tenkan > Kijun and price above cloud in bull regime with volume > 1.5 * 6h volume MA(20);
         Short when Tenkan < Kijun and price below cloud in bear regime with volume > 1.5 * 6h volume MA(20).
- Exit: Opposite Ichimoku cross (Tenkan/Kijun reversal) or price breaks opposite cloud edge.
- Signal size: 0.25 discrete to balance capture and fee control.
- Ichimoku provides dynamic support/resistance; EMA filter avoids counter-trend trades; volume confirms conviction.
- Works in bull (cloud support/respect) and bear (cloud resistance/rejection) due to adaptive cloud edges.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Ichimoku calculation and volume
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:
        return np.zeros(n)
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h Ichimoku components
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
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
    
    # Chikou Span (Lagging Span): close shifted 22 periods behind (not used for signals)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_6h, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_6h, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_b)
    
    # Calculate 6h volume MA(20) for confirmation
    volume_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(35, 52, 20, 1)  # EMA needs 35, Ichimoku needs 52, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or np.isnan(senkou_a_aligned[i]) or 
            np.isnan(senkou_b_aligned[i]) or np.isnan(vol_ma_6h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume confirmation: 1.5x threshold
        vol_confirm = curr_volume > 1.5 * vol_ma_6h_aligned[i]
        
        # Trend filter: price vs 1d EMA(34)
        bull_regime = curr_close > ema_34_1d_aligned[i]
        bear_regime = curr_close < ema_34_1d_aligned[i]
        
        # Cloud top and bottom
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Ichimoku cross signals
        tenkan_above_kijun = tenkan_aligned[i] > kijun_aligned[i]
        tenkan_below_kijun = tenkan_aligned[i] < kijun_aligned[i]
        
        if position == 0:
            # Check for entry signals
            # Long: Tenkan > Kijun AND price above cloud in bull regime with volume confirmation
            if (tenkan_above_kijun and curr_close > cloud_top and bull_regime and vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Tenkan < Kijun AND price below cloud in bear regime with volume confirmation
            elif (tenkan_below_kijun and curr_close < cloud_bottom and bear_regime and vol_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: exit on Tenkan/Kijun cross down OR price breaks below cloud bottom
            if (tenkan_below_kijun or curr_close < cloud_bottom):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on Tenkan/Kijun cross up OR price breaks above cloud top
            if (tenkan_above_kijun or curr_close > cloud_top):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0