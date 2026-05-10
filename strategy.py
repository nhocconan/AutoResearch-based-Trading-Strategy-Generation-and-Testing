# 4h_IchimokuKumo_CryptoTrend
# Hypothesis: Ichimoku Cloud (Tenkan/Kijun) with 4h price position relative to cloud and 1d trend filter for multi-timeframe confirmation.
# Long when price > cloud and Tenkan > Kijun in uptrend (price > 1d EMA50).
# Short when price < cloud and Tenkan < Kijun in downtrend (price < 1d EMA50).
# Uses volume confirmation (current 4h volume > 1.5x average 1d volume scaled) to reduce false breakouts.
# Ichimoku works in both bull and bear markets by capturing momentum shifts via cloud breaks and TK crosses.
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.

name = "4h_IchimokuKumo_CryptoTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_9 = np.full(n, np.nan)
    min_low_9 = np.full(n, np.nan)
    for i in range(n):
        if i >= period_tenkan - 1:
            start = i - period_tenkan + 1
            max_high_9[i] = np.max(high[start:i+1])
            min_low_9[i] = np.min(low[start:i+1])
    tenkan = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_26 = np.full(n, np.nan)
    min_low_26 = np.full(n, np.nan)
    for i in range(n):
        if i >= period_kijun - 1:
            start = i - period_kijun + 1
            max_high_26[i] = np.max(high[start:i+1])
            min_low_26[i] = np.min(low[start:i+1])
    kijun = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    max_high_52 = np.full(n, np.nan)
    min_low_52 = np.full(n, np.nan)
    for i in range(n):
        if i >= period_senkou_b - 1:
            start = i - period_senkou_b + 1
            max_high_52[i] = np.max(high[start:i+1])
            min_low_52[i] = np.min(low[start:i+1])
    senkou_b = (max_high_52 + min_low_52) / 2
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d volume SMA20 for volume confirmation
    volume_1d = df_1d['volume'].values
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period_kijun, period_senkou_b, 50)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        if np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or \
           np.isnan(senkou_b[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine if price is above or below cloud
        # Cloud top is max(senkou_a, senkou_b), bottom is min(senkou_a, senkou_b)
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # TK cross: Tenkan > Kijun (bullish) or Tenkan < Kijun (bearish)
        tk_bullish = tenkan[i] > kijun[i]
        tk_bearish = tenkan[i] < kijun[i]
        
        # Volume confirmation: current 4h volume > 1.5x average 1d volume (scaled)
        # 1d has 6 periods of 4h, so scale 1d volume by 1/6 to get equivalent 4h average
        vol_1d_scaled = vol_sma20_1d_aligned[i] / 6.0
        volume_confirm = volume[i] > 1.5 * vol_1d_scaled
        
        # Trend determination: price vs 1d EMA50
        is_uptrend = close[i] > ema50_1d_aligned[i]
        is_downtrend = close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # Long: price above cloud, TK bullish, in uptrend with volume
            if price_above_cloud and tk_bullish and is_uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price below cloud, TK bearish, in downtrend with volume
            elif price_below_cloud and tk_bearish and is_downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price falls below cloud or TK turns bearish
            if not price_above_cloud or not tk_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price rises above cloud or TK turns bullish
            if not price_below_cloud or not tk_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals