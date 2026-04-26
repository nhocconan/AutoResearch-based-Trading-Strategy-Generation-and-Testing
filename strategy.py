#!/usr/bin/env python3
"""
1h_Ichimoku_Cloud_Breakout_4hTrend_1dVolRegime
Hypothesis: On 1h timeframe, use Ichimoku cloud breakout (Tenkan/Kijun cross + price above/below cloud) for entry timing, 
with 4h EMA50 trend filter for signal direction and 1d volume regime filter (volume > 1.5x 20-period MA) to avoid chop.
Designed for low overtrading (target 15-30 trades/year) by requiring confluence of multiple timeframes and volume confirmation.
Works in bull/bear via 4h trend alignment and volume regime filter that adapts to volatility.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    # Trend: 1 = uptrend (close > EMA50), -1 = downtrend (close < EMA50)
    trend_4h = np.where(ema_50_4h_aligned > 0, 
                        np.where(close > ema_50_4h_aligned, 1, -1), 
                        0)
    
    # Load 1d data ONCE before loop for volume regime filter
    df_1d = get_htf_data(prices, '1d')
    # Calculate 1d volume MA20 for regime filter
    volume_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20, adjust=False).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    # Volume regime: 1 = high volume (above 1.5x MA), 0 = normal/low volume
    volume_regime = np.where(volume_ma_20_1d_aligned > 0, 
                             np.where(df_1d['volume'].values > (1.5 * volume_ma_20_1d_aligned), 1, 0), 
                             0)
    volume_regime_aligned = align_htf_to_ltf(prices, df_1d, volume_regime)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): close shifted 26 periods behind (not used for entry)
    
    # Cloud top and bottom (Senkou Span A and B)
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 52 for Senkou B, 26 for cloud shift, 50 for 4h EMA)
    start_idx = max(52, 26, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_regime_aligned[i]) or
            np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Ichimoku breakout conditions with 4h trend and 1d volume regime filters
        if position == 0:
            # Long: Price above cloud AND Tenkan crosses above Kijun AND 4h uptrend AND high volume regime
            if (close[i] > cloud_top[i] and tenkan[i] > kijun[i] and 
                trend_4h[i] == 1 and volume_regime_aligned[i] == 1):
                signals[i] = 0.20
                position = 1
            # Short: Price below cloud AND Tenkan crosses below Kijun AND 4h downtrend AND high volume regime
            elif (close[i] < cloud_bottom[i] and tenkan[i] < kijun[i] and 
                  trend_4h[i] == -1 and volume_regime_aligned[i] == 1):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: Price falls below cloud OR Tenkan crosses below Kijun OR 4h trend turns down
            if (close[i] < cloud_bottom[i] or tenkan[i] < kijun[i] or trend_4h[i] == -1):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: Price rises above cloud OR Tenkan crosses above Kijun OR 4h trend turns up
            if (close[i] > cloud_top[i] or tenkan[i] > kijun[i] or trend_4h[i] == 1):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Ichimoku_Cloud_Breakout_4hTrend_1dVolRegime"
timeframe = "1h"
leverage = 1.0