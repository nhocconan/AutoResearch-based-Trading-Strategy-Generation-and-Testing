#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation
# Ichimoku (Tenkan/Kijun cross + price vs cloud) provides multi-factor momentum signals.
# 1d EMA50 ensures alignment with higher timeframe trend to avoid counter-trend trades.
# Volume spike (>1.5x 20 EMA) confirms institutional participation.
# Discrete sizing 0.25 limits risk. Works in bull/bear: trend filter + cloud avoids whipsaws.
# Target: 50-150 trades over 4 years (12-37/year) on BTC/ETH.

name = "6h_Ichimoku_1dEMA50_VolumeSpike"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend direction
    close_1d = pd.Series(df_1d['close'])
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe (completed 1d bar only)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Ichimoku components on 6h timeframe
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_a = ((tenkan + kijun) / 2.0)
    senkou_b = ((period52_high + period52_low) / 2.0)
    
    # For cloud analysis, we need current Senkou spans (shifted back 26 periods to align with price)
    # Since we can't use future data, we use the values that were calculated 26 periods ago
    senkou_a_lag = senkou_a.shift(26) if hasattr(senkou_a, 'shift') else np.roll(senkou_a, 26)
    senkou_b_lag = senkou_b.shift(26) if hasattr(senkou_b, 'shift') else np.roll(senkou_b, 26)
    # Handle NaN for rolled values
    senkou_a_lag[:26] = np.nan
    senkou_b_lag[:26] = np.nan
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_aligned[i]) or np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a_lag[i]) or np.isnan(senkou_b_lag[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        # Cloud top and bottom
        cloud_top = max(senkou_a_lag[i], senkou_b_lag[i])
        cloud_bottom = min(senkou_a_lag[i], senkou_b_lag[i])
        
        if position == 0:
            # Long conditions: price above cloud + Tenkan > Kijun + uptrend + volume spike
            if (close[i] > cloud_top and tenkan[i] > kijun[i] and 
                close[i] > ema50_aligned[i] and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short conditions: price below cloud + Tenkan < Kijun + downtrend + volume spike
            elif (close[i] < cloud_bottom and tenkan[i] < kijun[i] and 
                  close[i] < ema50_aligned[i] and volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price drops below cloud OR Tenkan < Kijun OR trend changes
            if (close[i] < cloud_top or tenkan[i] < kijun[i] or 
                close[i] < ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above cloud OR Tenkan > Kijun OR trend changes
            if (close[i] > cloud_bottom or tenkan[i] > kijun[i] or 
                close[i] > ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals