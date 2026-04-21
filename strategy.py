# AUTO-GENERATED: strategy.py
#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout
Hypothesis: Uses Ichimoku Cloud on 1d timeframe as trend filter and support/resistance, 
with Tenkan-Kijun cross on 6h for entry timing. Enters long when price is above 
1d cloud and TK crosses bullish; short when below cloud and TK crosses bearish. 
Adds volume confirmation to avoid false breaks. Works in bull markets by riding 
uptrends above cloud and in bear markets by shorting bounces below cloud. 
Target: 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # === Ichimoku Components on 1d ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    max_high_9 = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    max_high_26 = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period_senkou_b = 52
    max_high_52 = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (max_high_52 + min_low_52) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_6h, senkou_b_6h)
    cloud_bottom = np.minimum(senkou_a_6h, senkou_b_6h)
    
    # === 6h TK Cross ===
    close_6h = prices['close'].values
    # Tenkan-sen on 6h (9-period)
    max_high_9_6h = pd.Series(close_6h).rolling(window=9, min_periods=9).max().values
    min_low_9_6h = pd.Series(close_6h).rolling(window=9, min_periods=9).min().values
    tenkan_6h_raw = (max_high_9_6h + min_low_9_6h) / 2
    
    # Kijun-sen on 6h (26-period)
    max_high_26_6h = pd.Series(close_6h).rolling(window=26, min_periods=26).max().values
    min_low_26_6h = pd.Series(close_6h).rolling(window=26, min_periods=26).min().values
    kijun_6h_raw = (max_high_26_6h + min_low_26_6h) / 2
    
    # TK cross signals
    tk_cross_bull = (tenkan_6h_raw > kijun_6h_raw) & (np.roll(tenkan_6h_raw, 1) <= np.roll(kijun_6h_raw, 1))
    tk_cross_bear = (tenkan_6h_raw < kijun_6h_raw) & (np.roll(tenkan_6h_raw, 1) >= np.roll(kijun_6h_raw, 1))
    
    # === Volume Confirmation ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after warmup for Senkou B
        # Skip if indicators not ready
        if (np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or 
            np.isnan(tk_cross_bull[i]) or np.isnan(tk_cross_bear[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close_6h[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Long: price above cloud + bullish TK cross + volume
            if (price_close > cloud_top[i] and
                tk_cross_bull[i] and
                vol_ratio_val > 1.2):
                signals[i] = 0.25
                position = 1
            # Short: price below cloud + bearish TK cross + volume
            elif (price_close < cloud_bottom[i] and
                  tk_cross_bear[i] and
                  vol_ratio_val > 1.2):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price crosses back into cloud or opposite TK cross
            if position == 1:
                if (price_close < cloud_top[i] or tk_cross_bear[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (price_close > cloud_bottom[i] or tk_cross_bull[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout"
timeframe = "6h"
leverage = 1.0