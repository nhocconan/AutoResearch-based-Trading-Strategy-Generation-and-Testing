#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d Trend Filter and Volume Confirmation
# Uses Ichimoku cloud (Tenkan/Kijun) on 6h for entry signals, with 1d EMA50 trend filter.
# Long when Tenkan crosses above Kijun, price above cloud, and 1d EMA50 uptrend.
# Short when Tenkan crosses below Kijun, price below cloud, and 1d EMA50 downtrend.
# Volume confirmation requires current volume > 1.5x 20-period average.
# Targets 60-120 total trades over 4 years (15-30/year) with clear entry/exit rules.
# Ichimoku provides dynamic support/resistance; higher timeframe filter avoids counter-trend trades.

name = "6h_Ichimoku_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === Ichimoku Components (9, 26, 52 periods) on 6h data ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # Kumo (Cloud) top and bottom (Senkou Span shifted 26 periods ahead)
    # For simplicity, we use current Senkou spans as cloud boundaries
    # In practice, cloud is plotted 26 periods ahead, but for filtering we check if price is above/both spans
    kumo_top = np.maximum(senkou_a, senkou_b)
    kumo_bottom = np.minimum(senkou_a, senkou_b)
    
    # === 1d EMA50 for trend filter ===
    daily_close = df_1d['close'].values
    ema_50 = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # === Volume confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after warmup for Ichimoku
        # Get values
        close_val = close[i]
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        tenkan_prev = tenkan[i-1]
        kijun_prev = kijun[i-1]
        kumo_top_val = kumo_top[i]
        kumo_bottom_val = kumo_bottom[i]
        ema_val = ema_50_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(tenkan_val) or np.isnan(kijun_val) or np.isnan(tenkan_prev) or 
            np.isnan(kijun_prev) or np.isnan(kumo_top_val) or np.isnan(kumo_bottom_val) or 
            np.isnan(ema_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Tenkan crosses above Kijun, price above cloud, 1d uptrend, volume confirmation
            if (tenkan_prev <= kijun_prev and tenkan_val > kijun_val and 
                close_val > kumo_top_val and close_val > ema_val and vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Short entry: Tenkan crosses below Kijun, price below cloud, 1d downtrend, volume confirmation
            elif (tenkan_prev >= kijun_prev and tenkan_val < kijun_val and 
                  close_val < kumo_bottom_val and close_val < ema_val and vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Tenkan crosses below Kijun or price falls below cloud
            if tenkan_val < kijun_val or close_val < kumo_top_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Tenkan crosses above Kijun or price rises above cloud
            if tenkan_val > kijun_val or close_val > kumo_bottom_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals