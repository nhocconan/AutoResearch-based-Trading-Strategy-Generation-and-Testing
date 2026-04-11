#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud + Volume Confirmation
# - Uses Ichimoku components from 1d timeframe for trend direction (more stable than 6h)
# - Tenkan-sen (9-period) and Kijun-sen (26-period) cross for momentum signals
# - Senkou Span A/B form the cloud for trend filtration
# - Chikou Span (lagging 26 periods) confirms price action
# - Long: Price > Cloud AND Tenkan > Kijun AND Chikou > Price(26 periods ago) AND volume > 1.5x 20-period average
# - Short: Price < Cloud AND Tenkan < Kijun AND Chikou < Price(26 periods ago) AND volume > 1.5x 20-period average
# - Ichimoku works in both bull and bear markets by adapting to trend context
# - Volume confirmation filters weak breakouts
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits

name = "6h_1d_ichimoku_cloud_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return signals
    
    # Pre-compute Ichimoku components on 1d timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close plotted 26 periods behind
    chikou_span = close_1d  # Will be aligned with proper offset
    
    # Align all Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a, additional_delay_bars=26)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b, additional_delay_bars=26)
    chikou_aligned = align_htf_to_ltf(prices, df_1d, chikou_span, additional_delay_bars=26)
    
    # Pre-compute 1d volume confirmation (20-period average)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(chikou_aligned[i]) or np.isnan(volume_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price = close[i]
        volume_current = volume[i]
        
        # Ichimoku trend conditions
        # Cloud top/bottom (Senkou Span A/B form the cloud)
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Price relative to cloud
        price_above_cloud = price > cloud_top
        price_below_cloud = price < cloud_bottom
        
        # Tenkan/Kijun cross
        tenkan_above_kijun = tenkan_aligned[i] > kijun_aligned[i]
        tenkan_below_kijun = tenkan_aligned[i] < kijun_aligned[i]
        
        # Chikou confirmation (price 26 periods ago vs current Chikou)
        chikou_price = close[i - 26] if i >= 26 else np.nan
        chikou_confirm_bull = not np.isnan(chikou_price) and chikou_aligned[i] > chikou_price
        chikou_confirm_bear = not np.isnan(chikou_price) and chikou_aligned[i] < chikou_price
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Price above cloud + bullish TK cross + bullish Chikou + volume
        if price_above_cloud and tenkan_above_kijun and chikou_confirm_bull and vol_confirm:
            enter_long = True
        
        # Short: Price below cloud + bearish TK cross + bearish Chikou + volume
        if price_below_cloud and tenkan_below_kijun and chikou_confirm_bear and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite signals or cloud reversal
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price falls below cloud OR bearish TK cross
            exit_long = price_below_cloud or tenkan_below_kijun
        elif position == -1:
            # Exit short if price rises above cloud OR bullish TK cross
            exit_short = price_above_cloud or tenkan_above_kijun
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals