#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud + TK Cross + 1d Volume Confirmation
# Ichimoku provides multi-component trend system: Tenkan/Kijun cross for momentum,
# Cloud (Senkou Span A/B) for dynamic support/resistance, and price/cloud relationship for trend bias.
# Only trade TK crosses in direction of 1d trend (price vs 1d EMA50) to avoid whipsaws.
# Volume confirmation (1.8x 24-period average) ensures institutional participation.
# Works in bull markets via buying TK crosses above cloud in uptrends,
# and in bear markets via selling TK crosses below cloud in downtrends.
# Targets 12-37 trades/year (50-150 total over 4 years) for 6h timeframe.

name = "6h_Ichimoku_TK_Cross_1dTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # TK Cross signals
    tk_cross_above = (tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1))
    tk_cross_below = (tenkan < kijun) & (tenkan.shift(1) >= kijun.shift(1))
    
    # Price vs Cloud relationship for trend bias
    # Price above Cloud: bullish, Price below Cloud: bearish
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    # Volume confirmation (1.8x 24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Ichimoku calculations)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: TK cross above + price above cloud + 1d uptrend + volume spike
            if (tk_cross_above[i] and price_above_cloud[i] and 
                close[i] > ema_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: TK cross below + price below cloud + 1d downtrend + volume spike
            elif (tk_cross_below[i] and price_below_cloud[i] and 
                  close[i] < ema_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: TK cross below OR price breaks below cloud OR 1d trend breaks
            if tk_cross_below[i] or close[i] < cloud_top[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: TK cross above OR price breaks above cloud OR 1d trend breaks
            if tk_cross_above[i] or close[i] > cloud_bottom[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals