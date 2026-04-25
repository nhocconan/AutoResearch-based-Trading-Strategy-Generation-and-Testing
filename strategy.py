#!/usr/bin/env python3
"""
6h_IchimokuCloud_Breakout_1dTrend_VolumeConfirm
Hypothesis: 6h Ichimoku cloud breakout with 1d trend filter (price > 1d EMA50) and volume confirmation.
Long when Tenkan-sen > Kijun-sen AND price breaks above Kumo cloud top in 1d uptrend with volume > 1.8x 20-period average.
Short when Tenkan-sen < Kijun-sen AND price breaks below Kumo cloud bottom in 1d downtrend with volume > 1.8x 20-period average.
Exit when price re-enters Kumo cloud or Tenkan/Kijun cross reverses.
Designed for ~25-35 trades/year by requiring Ichimoku alignment, trend filter, and volume spike.
Works in bull/bear markets via 1d EMA50 filter; Ichimoku provides dynamic support/resistance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen: (9-period high + 9-period low) / 2
    period_tenkan = 9
    highest_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    lowest_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (highest_tenkan + lowest_tenkan) / 2
    
    # Kijun-sen: (26-period high + 26-period low) / 2
    period_kijun = 26
    highest_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    lowest_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (highest_kijun + lowest_kijun) / 2
    
    # Senkou Span A: (Tenkan + Kijun) / 2 plotted 26 periods ahead
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B: (52-period high + 52-period low) / 2 plotted 26 periods ahead
    period_senkou_b = 52
    highest_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    lowest_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (highest_senkou_b + lowest_senkou_b) / 2
    
    # Kumo cloud top/bottom (current)
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    
    # Volume regime: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations (max of all periods + 26 for Senkou shift)
    start_idx = max(100, period_kijun, period_senkou_b) + 26
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_1d_aligned[i]
        
        if position == 0:
            # Only trade in trending regimes (1d EMA50 filter)
            if close[i] > ema_trend:  # 1d uptrend regime
                # Long: Tenkan > Kijun AND price breaks above cloud top with volume
                long_signal = (tenkan[i] > kijun[i]) and (close[i] > cloud_top[i]) and vol_regime[i]
            else:  # 1d downtrend regime
                # Short: Tenkan < Kijun AND price breaks below cloud bottom with volume
                short_signal = (tenkan[i] < kijun[i]) and (close[i] < cloud_bottom[i]) and vol_regime[i]
            
            if 'long_signal' in locals() and long_signal:
                signals[i] = 0.25
                position = 1
            elif 'short_signal' in locals() and short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
                # Clear signal variables for next iteration
                if 'long_signal' in locals(): del long_signal
                if 'short_signal' in locals(): del short_signal
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price re-enters cloud OR Tenkan/Kijun cross reverses (Tenkan < Kijun)
            if close[i] <= cloud_top[i] or tenkan[i] < kijun[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters cloud OR Tenkan/Kijun cross reverses (Tenkan > Kijun)
            if close[i] >= cloud_bottom[i] or tenkan[i] > kijun[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_IchimokuCloud_Breakout_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0