#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation
# Long when: Tenkan-sen crosses above Kijun-sen AND price above 1d EMA50 AND volume > 1.5x 20-period MA AND price above Kumo cloud
# Short when: Tenkan-sen crosses below Kijun-sen AND price below 1d EMA50 AND volume > 1.5x 20-period MA AND price below Kumo cloud
# Exit when: Tenkan-sen/Kijun-sen cross reverses OR price crosses Kumo cloud in opposite direction
# Uses Ichimoku for momentum/trend structure, 1d EMA for higher timeframe trend, volume for conviction
# Timeframe: 6h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_Ichimoku_Cloud_1dEMA50_VolumeConfirm"
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
    
    # Calculate volume confirmation on 6h using 20-period MA
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Calculate Ichimoku components on 6h (9, 26, 52 periods)
    if len(high) >= 52 and len(low) >= 52 and len(close) >= 52:
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
        period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
        period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
        tenkan_sen = (period9_high + period9_low) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
        period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
        period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
        kijun_sen = (period26_high + period26_low) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
        senkou_span_a = (tenkan_sen + kijun_sen) / 2
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
        period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
        period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
        senkou_span_b = (period52_high + period52_low) / 2
        
        # Kumo cloud boundaries
        upper_kumo = np.maximum(senkou_span_a, senkou_span_b)
        lower_kumo = np.minimum(senkou_span_a, senkou_span_b)
        
        # Tenkan/Kijun cross signals
        tenkan_prev = np.roll(tenkan_sen, 1)
        tenkan_prev[0] = np.nan
        kijun_prev = np.roll(kijun_sen, 1)
        kijun_prev[0] = np.nan
        tk_cross_above = (tenkan_prev <= kijun_prev) & (tenkan_sen > kijun_sen)
        tk_cross_below = (tenkan_prev >= kijun_prev) & (tenkan_sen < kijun_sen)
        
        # Price relative to cloud
        price_above_kumo = close > upper_kumo
        price_below_kumo = close < lower_kumo
    else:
        tenkan_sen = np.full(n, np.nan)
        kijun_sen = np.full(n, np.nan)
        senkou_span_a = np.full(n, np.nan)
        senkou_span_b = np.full(n, np.nan)
        upper_kumo = np.full(n, np.nan)
        lower_kumo = np.full(n, np.nan)
        tk_cross_above = np.zeros(n, dtype=bool)
        tk_cross_below = np.zeros(n, dtype=bool)
        price_above_kumo = np.zeros(n, dtype=bool)
        price_below_kumo = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on 1d timeframe
    if len(close_1d) >= 50:
        ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
        ema_above = close_1d > ema_50_1d
        ema_below = close_1d < ema_50_1d
    else:
        ema_above = np.full(len(close_1d), False)
        ema_below = np.full(len(close_1d), False)
    
    # Align 1d EMA trend to 6h timeframe
    ema_above_aligned = align_htf_to_ltf(prices, df_1d, ema_above.astype(float))
    ema_below_aligned = align_htf_to_ltf(prices, df_1d, ema_below.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or np.isnan(upper_kumo[i]) or 
            np.isnan(lower_kumo[i]) or np.isnan(ema_above_aligned[i]) or np.isnan(ema_below_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: TK cross above + price above 1d EMA50 + volume filter + price above Kumo
            if (tk_cross_above[i] and 
                ema_above_aligned[i] == 1.0 and 
                volume_filter[i] and 
                price_above_kumo[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: TK cross below + price below 1d EMA50 + volume filter + price below Kumo
            elif (tk_cross_below[i] and 
                  ema_below_aligned[i] == 1.0 and 
                  volume_filter[i] and 
                  price_below_kumo[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TK cross below OR price crosses below Kumo
            if (tk_cross_below[i] or (close[i] < lower_kumo[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TK cross above OR price crosses above Kumo
            if (tk_cross_above[i] or (close[i] > upper_kumo[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals