#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation.
# Uses Tenkan-sen (9), Kijun-sen (26), Senkou Span A/B (26, 52 displacement) from 6h data.
# Long: Price > Cloud AND Tenkan > Kijun (bullish TK cross) AND price > 1d EMA50 (uptrend) AND volume > 2x 20-period MA
# Short: Price < Cloud AND Tenkan < Kijun (bearish TK cross) AND price < 1d EMA50 (downtrend) AND volume > 2x 20-period MA
# Exit: Opposite TK cross or price crosses Cloud mid-point.
# Ichimoku provides dynamic support/resistance via Cloud; TK cross signals momentum shift;
# 1d EMA50 filters higher timeframe trend; volume confirmation reduces whipsaws.
# Works in bull via long signals with trend alignment and in bear via short signals with trend alignment.
# Target: 80-160 total trades over 4 years (20-40/year).

name = "6h_Ichimoku_1dEMA50_Volume"
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku calculations (using 6h data)
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
    
    # The Cloud (Kumo) is between Senkou Span A and Senkou Span B
    # For simplicity, we use the current values (no shift in calculation here as alignment handles timing)
    # Upper Cloud: max(Senkou A, Senkou B)
    # Lower Cloud: min(Senkou A, Senkou B)
    upper_cloud = np.maximum(senkou_a, senkou_b)
    lower_cloud = np.minimum(senkou_a, senkou_b)
    
    # Cloud Mid-point: (Upper Cloud + Lower Cloud)/2
    cloud_mid = (upper_cloud + lower_cloud) / 2
    
    # TK Cross: Tenkan > Kijun (bullish), Tenkan < Kijun (bearish)
    tk_bullish = tenkan > kijun
    tk_bearish = tenkan < kijun
    
    # Price vs Cloud: Price > Upper Cloud (bullish), Price < Lower Cloud (bearish)
    price_above_cloud = close > upper_cloud
    price_below_cloud = close < lower_cloud
    
    # Volume regime: current 6h volume > 2x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_1d_aligned[i]
        tk_bull = tk_bullish[i]
        tk_bear = tk_bearish[i]
        price_above = price_above_cloud[i]
        price_below = price_below_cloud[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_uptrend = close_val > ema_trend
        is_downtrend = close_val < ema_trend
        
        # Entry logic
        if position == 0:
            # Long: Price > Cloud AND Tenkan > Kijun (bullish TK cross) AND uptrend AND volume spike
            if price_above and tk_bull and is_uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Price < Cloud AND Tenkan < Kijun (bearish TK cross) AND downtrend AND volume spike
            elif price_below and tk_bear and is_downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price < Cloud Mid-point OR bearish TK cross OR trend turns down
            if close_val < cloud_mid[i] or tk_bear or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price > Cloud Mid-point OR bullish TK cross OR trend turns up
            if close_val > cloud_mid[i] or tk_bull or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals