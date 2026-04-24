#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w EMA50 for major trend filter (defines bull/bear regime).
- Entry: Long when price is above Ichimoku cloud AND Tenkan > Kijun in bull regime with volume > 1.5 * 6h volume MA(20);
         Short when price is below Ichimoku cloud AND Tenkan < Kijun in bear regime with volume > 1.5 * 6h volume MA(20).
- Exit: Opposite Ichimoku cloud cross (price crosses cloud in opposite direction).
- Signal size: 0.25 discrete to balance capture and fee control.
- Ichimoku provides dynamic support/resistance; weekly EMA filters major trend; volume confirms conviction.
- Works in bull (trend alignment) and bear (regime-aware mean reversion at cloud edges).
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
    if len(df_6h) < 52:  # Ichimoku needs up to 52 periods
        return np.zeros(n)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close']
    ema_50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(df_6h['high']).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(df_6h['low']).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(df_6h['high']).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(df_6h['low']).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(df_6h['high']).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(df_6h['low']).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close plotted 26 periods behind
    # Not used for signals as it requires future data
    
    # Align Ichimoku components to 6h timeframe (no shift needed as they are already plotted forward/backward)
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
    # Ichimoku needs 52 periods for Senkou B, plus alignment considerations
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(vol_ma_6h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Determine cloud boundaries (Senkou Span A and B)
        upper_cloud = max(senkou_a_aligned[i], senkou_b_aligned[i])
        lower_cloud = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Volume confirmation: 1.5x threshold
        vol_confirm = curr_volume > 1.5 * vol_ma_6h_aligned[i]
        
        # Trend filter: 1w EMA50
        bull_regime = ema50_aligned[i] > 0 and close[i] > ema50_aligned[i]  # Price above weekly EMA50
        bear_regime = ema50_aligned[i] > 0 and close[i] < ema50_aligned[i]  # Price below weekly EMA50
        
        if position == 0:
            # Check for entry signals
            # Long: price above cloud AND Tenkan > Kijun in bull regime with volume confirmation
            if (curr_close > upper_cloud and tenkan_aligned[i] > kijun_aligned[i] and 
                bull_regime and vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price below cloud AND Tenkan < Kijun in bear regime with volume confirmation
            elif (curr_close < lower_cloud and tenkan_aligned[i] < kijun_aligned[i] and 
                  bear_regime and vol_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: exit when price crosses below cloud (bearish cloud break)
            if curr_close < lower_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price crosses above cloud (bullish cloud break)
            if curr_close > upper_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0