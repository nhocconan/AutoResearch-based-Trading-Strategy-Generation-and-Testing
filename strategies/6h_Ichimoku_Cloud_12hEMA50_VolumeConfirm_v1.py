#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 12h trend filter and volume confirmation
# Long when price > cloud AND Tenkan > Kijun AND price > 12h EMA50 AND volume > 1.5x 20-bar avg
# Short when price < cloud AND Tenkan < Kijun AND price < 12h EMA50 AND volume > 1.5x 20-bar avg
# Exit when price crosses Tenkan-Kijun line or opposite cloud boundary
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 12-30 trades/year on 6h.
# Ichimoku provides dynamic support/resistance and trend direction. 12h EMA50 filters counter-trend moves.
# Volume confirmation ensures institutional participation. Strategy avoids overtrading with strict confluence.

name = "6h_Ichimoku_Cloud_12hEMA50_VolumeConfirm_v1"
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate EMA(50) on 12h data
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Ichimoku Cloud components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2.0
    
    # The cloud is between senkou_a and senkou_b
    # Upper cloud boundary: max(senkou_a, senkou_b)
    # Lower cloud boundary: min(senkou_a, senkou_b)
    upper_cloud = np.maximum(senkou_a, senkou_b)
    lower_cloud = np.minimum(senkou_a, senkou_b)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 50, 20)  # Ichimoku(52), EMA50, volume MA all need warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(upper_cloud[i]) or 
            np.isnan(lower_cloud[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        ema_50 = ema_50_12h_aligned[i]
        
        # Ichimoku conditions
        price_above_cloud = curr_close > upper_cloud[i]
        price_below_cloud = curr_close < lower_cloud[i]
        tenkan_above_kijun = tenkan[i] > kijun[i]
        tenkan_below_kijun = tenkan[i] < kijun[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price < Tenkan-Kijun midpoint OR price < lower cloud
            exit_condition = (curr_close < (tenkan[i] + kijun[i]) / 2.0) or (curr_close < lower_cloud[i])
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Tenkan-Kijun midpoint OR price > upper cloud
            exit_condition = (curr_close > (tenkan[i] + kijun[i]) / 2.0) or (curr_close > upper_cloud[i])
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price > cloud AND Tenkan > Kijun AND price > 12h EMA50 AND volume confirmation
            if price_above_cloud and tenkan_above_kijun and curr_close > ema_50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price < cloud AND Tenkan < Kijun AND price < 12h EMA50 AND volume confirmation
            elif price_below_cloud and tenkan_below_kijun and curr_close < ema_50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals