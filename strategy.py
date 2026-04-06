#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku cloud breakout with 1d trend filter and volume confirmation
# Long when price breaks above Ichimoku cloud (Tenkan-sen > Kijun-sen AND price > Senkou Span A/B) AND price > 1d EMA(50) AND volume > 2x 20-period average
# Short when price breaks below Ichimoku cloud (Tenkan-sen < Kijun-sen AND price < Senkou Span A/B) AND price < 1d EMA(50) AND volume > 2x 20-period average
# Exit when price crosses back into the cloud or Tenkan/Kijun cross reverses
# Uses Ichimoku for trend/momentum, 1d EMA for higher timeframe trend filter, volume for confirmation
# Target: 50-150 total trades over 4 years (12-37/year) for optimal 6h performance

name = "6h_ichimoku_1d_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku Cloud Components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_span_b = (period52_high + period52_low) / 2
    
    # 1-day EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    
    # Calculate 50-period EMA on daily close
    daily_close_series = pd.Series(daily_close)
    daily_ema = daily_close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align daily EMA to 6h timeframe
    daily_ema_aligned = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 2.0 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after longest lookback
        # Skip if required data not available
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or 
            np.isnan(daily_ema_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A/B shifted forward 26 periods)
        # For current price, we use Senkou Span values from 26 periods ago
        if i >= 26:
            span_a = senkou_span_a[i-26]
            span_b = senkou_span_b[i-26]
        else:
            # Not enough data for cloud projection
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Cloud top and bottom
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        # Check exits: price re-enters cloud or TK cross reverses
        if position == 1:  # long position
            if (close[i] <= cloud_top and close[i] >= cloud_bottom) or \
               (tenkan_sen[i] < kijun_sen[i] and tenkan_sen[i-1] >= kijun_sen[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if (close[i] <= cloud_top and close[i] >= cloud_bottom) or \
               (tenkan_sen[i] > kijun_sen[i] and tenkan_sen[i-1] <= kijun_sen[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend filter and volume confirmation
            # Bullish TK cross: Tenkan crosses above Kijun
            tk_bullish = (tenkan_sen[i] > kijun_sen[i] and tenkan_sen[i-1] <= kijun_sen[i-1])
            # Bearish TK cross: Tenkan crosses below Kijun
            tk_bearish = (tenkan_sen[i] < kijun_sen[i] and tenkan_sen[i-1] >= kijun_sen[i-1])
            
            # Price above/below cloud
            price_above_cloud = close[i] > cloud_top
            price_below_cloud = close[i] < cloud_bottom
            
            # Long: bullish TK cross AND price above cloud AND price > daily EMA AND volume confirmation
            if (tk_bullish and price_above_cloud and 
                close[i] > daily_ema_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish TK cross AND price below cloud AND price < daily EMA AND volume confirmation
            elif (tk_bearish and price_below_cloud and 
                  close[i] < daily_ema_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals