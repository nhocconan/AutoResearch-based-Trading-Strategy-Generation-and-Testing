#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d TK Cross and Volume Confirmation
# Uses Ichimoku system: Tenkan-sen (9), Kijun-sen (26), Senkou Span A/B (26/52), Chikou Span (26)
# Long when: price > cloud, TK cross bullish, and 1d trend up (close > EMA50)
# Short when: price < cloud, TK cross bearish, and 1d trend down (close < EMA50)
# Volume filter: current volume > 1.5x 20-period average
# Ichimoku provides dynamic support/resistance; 1d EMA50 filters higher timeframe trend.
# Target: 60-120 total trades over 4 years (15-30/year).

name = "6h_Ichimoku_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === Ichimoku Components (6h timeframe) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): close shifted 26 periods back
    chikou = pd.Series(close).shift(26)
    
    # === Daily EMA50 for trend filter ===
    daily_close = df_1d['close'].values
    ema_50 = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # === Volume confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(52, n):  # Start after warmup for Senkou Span B
        # Get values
        close_val = close[i]
        tenkan_val = tenkan.iloc[i] if hasattr(tenkan, 'iloc') else tenkan[i]
        kijun_val = kijun.iloc[i] if hasattr(kijun, 'iloc') else kijun[i]
        senkou_a_val = senkou_a.iloc[i] if hasattr(senkou_a, 'iloc') else senkou_a[i]
        senkou_b_val = senkou_b.iloc[i] if hasattr(senkou_b, 'iloc') else senkou_b[i]
        chikou_val = chikou.iloc[i] if hasattr(chikou, 'iloc') else chikou[i]
        ema_val = ema_50_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(tenkan_val) or np.isnan(kijun_val) or np.isnan(senkou_a_val) or 
            np.isnan(senkou_b_val) or np.isnan(chikou_val) or np.isnan(ema_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud boundaries (Senkou Span A/B)
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        # TK cross conditions
        tk_bullish = tenkan_val > kijun_val
        tk_bearish = tenkan_val < kijun_val
        
        if position == 0:
            # Long entry: price above cloud, TK bullish, 1d uptrend, volume confirmation
            if (close_val > cloud_top and tk_bullish and 
                close_val > ema_val and vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short entry: price below cloud, TK bearish, 1d downtrend, volume confirmation
            elif (close_val < cloud_bottom and tk_bearish and 
                  close_val < ema_val and vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        
        elif position == 1:
            # Long exit: price drops below cloud or TK turns bearish
            if close_val < cloud_bottom or not tk_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above cloud or TK turns bullish
            if close_val > cloud_top or tk_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals