#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d volume filter
# - Primary: 6h price breaking above/below Ichimoku Cloud (Senkou Span A/B) from 1d
# - HTF: 1d volume confirmation (current volume > 1.3x 20-period MA) to avoid false breakouts
# - Long: 6h close > Senkou Span A AND Senkou Span B + volume confirmation
# - Short: 6h close < Senkou Span A AND Senkou Span B + volume confirmation
# - Exit: Price returns to Cloud midpoint (Senkou Span A/B average)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Ichimoku Cloud acts as dynamic support/resistance, volume confirms momentum
# - Target: 60-120 trades over 4 years (15-30/year) to stay within fee drag limits

name = "6h_1d_ichimoku_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 52 periods for Ichimoku (26*2)
        return np.zeros(n)
    
    # Pre-compute 6h data
    close_6h = prices['close'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Ichimoku Cloud components (1d)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2.0
    
    # Cloud midpoint (for exit): (Senkou Span A + Senkou Span B) / 2
    cloud_midpoint = (senkou_span_a + senkou_span_b) / 2.0
    
    # Align Ichimoku components to 6h timeframe
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    cloud_midpoint_aligned = align_htf_to_ltf(prices, df_1d, cloud_midpoint)
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after Ichimoku warmup period
        # Skip if any required data is invalid
        if (np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(cloud_midpoint_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume (aligned to 6h)
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        
        # Volume confirmation: current 1d volume > 1.3x 20-period MA
        volume_confirm = volume_1d_aligned[i] > 1.3 * volume_ma_20_1d_aligned[i]
        
        # Ichimoku breakout conditions
        # Cloud top is the higher of Senkou Span A and B
        cloud_top = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        # Cloud bottom is the lower of Senkou Span A and B
        cloud_bottom = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        breakout_long = close_6h[i] > cloud_top
        breakout_short = close_6h[i] < cloud_bottom
        
        # Exit condition: Price returns to cloud midpoint
        exit_long = close_6h[i] < cloud_midpoint_aligned[i]
        exit_short = close_6h[i] > cloud_midpoint_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Breakout above cloud + volume confirmation
            if breakout_long and volume_confirm:
                position = 1
                signals[i] = 0.25
            # Short entry: Breakout below cloud + volume confirmation
            elif breakout_short and volume_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price returns to cloud midpoint
            if position == 1:  # Long position
                if exit_long:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if exit_short:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals