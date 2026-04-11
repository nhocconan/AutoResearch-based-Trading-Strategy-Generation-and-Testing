#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation
# - Uses Ichimoku components (Tenkan-sen, Kijun-sen, Senkou Span A/B, Chikou Span) on 6h
# - Long: Price above cloud + TK cross bullish + 1d trend up (price > 1d Kijun-sen) + volume > 1.2x 20-period average
# - Short: Price below cloud + TK cross bearish + 1d trend down (price < 1d Kijun-sen) + volume > 1.2x 20-period average
# - Exit: TK cross in opposite direction or price re-enters cloud
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Ichimoku provides dynamic support/resistance and trend identification
# - 1d trend filter ensures alignment with higher timeframe momentum
# - Volume confirmation filters out weak signals

name = "6h_1d_ichimoku_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need sufficient data for Ichimoku calculations
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d Kijun-sen (base line) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    kijun_sen_1d = (pd.Series(high_1d).rolling(window=26, min_periods=26).max().values + 
                    pd.Series(low_1d).rolling(window=26, min_periods=26).min().values) / 2
    kijun_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen_1d)
    
    # Pre-compute 1d volume average for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute Ichimoku components on 6h timeframe
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    tenkan_sen = (pd.Series(high).rolling(window=9, min_periods=9).max().values + 
                  pd.Series(low).rolling(window=9, min_periods=9).min().values) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun_sen = (pd.Series(high).rolling(window=26, min_periods=26).max().values + 
                 pd.Series(low).rolling(window=26, min_periods=26).min().values) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    # Note: We don't actually shift forward here since align_htf_to_ltf handles timing
    # For cloud calculation, we use current values
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 shifted 26 periods ahead
    senkou_span_b = ((pd.Series(high).rolling(window=52, min_periods=52).max().values + 
                      pd.Series(low).rolling(window=52, min_periods=52).min().values) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    # We don't use Chikou Span in this strategy to avoid look-ahead issues
    
    # Calculate cloud boundaries (using current Senkou Span A/B)
    # The cloud is formed by Senkou Span A and B plotted 26 periods ahead
    # For current price, we compare with Senkou Span values from 26 periods ago
    # But to avoid look-ahead, we use the values that were known 26 periods ago
    # We'll calculate the cloud bottom/top as min/max of Senkou Span A/B
    senkou_span_a_shifted = np.roll(senkou_span_a, 26)
    senkou_span_b_shifted = np.roll(senkou_span_b, 26)
    # Initialize first 26 values
    senkou_span_a_shifted[:26] = senkou_span_a[:26]
    senkou_span_b_shifted[:26] = senkou_span_b[:26]
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_span_a_shifted, senkou_span_b_shifted)
    cloud_bottom = np.minimum(senkou_span_a_shifted, senkou_span_b_shifted)
    
    # TK Cross: Tenkan-sen crossing Kijun-sen
    tk_cross = tenkan_sen - kijun_sen
    tk_cross_prev = np.roll(tk_cross, 1)
    tk_cross_prev[0] = 0
    
    # Bullish TK cross: previous <= 0 and current > 0
    tk_bullish = (tk_cross_prev <= 0) & (tk_cross > 0)
    # Bearish TK cross: previous >= 0 and current < 0
    tk_bearish = (tk_cross_prev >= 0) & (tk_cross < 0)
    
    for i in range(100, n):  # Start after sufficient warmup
        # Skip if any required data is invalid
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or np.isnan(cloud_top[i]) or 
            np.isnan(cloud_bottom[i]) or np.isnan(kijun_sen_1d_aligned[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Ichimoku levels
        tenkan = tenkan_sen[i]
        kijun = kijun_sen[i]
        top_cloud = cloud_top[i]
        bottom_cloud = cloud_bottom[i]
        
        # 1d trend and volume confirmation
        trend_1d_up = close_price > kijun_sen_1d_aligned[i]
        trend_1d_down = close_price < kijun_sen_1d_aligned[i]
        vol_confirm = volume_current > 1.2 * volume_sma_20_1d_aligned[i]
        
        # TK cross signals
        bullish_tk = tk_bullish[i]
        bearish_tk = tk_bearish[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Price above cloud + bullish TK cross + 1d trend up + volume confirmation
        if (close_price > top_cloud and bullish_tk and trend_1d_up and vol_confirm):
            enter_long = True
        
        # Short: Price below cloud + bearish TK cross + 1d trend down + volume confirmation
        if (close_price < bottom_cloud and bearish_tk and trend_1d_down and vol_confirm):
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if bearish TK cross or price re-enters cloud (below top)
            exit_long = bearish_tk or (close_price < top_cloud)
        elif position == -1:
            # Exit short if bullish TK cross or price re-enters cloud (above bottom)
            exit_short = bullish_tk or (close_price > bottom_cloud)
        
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