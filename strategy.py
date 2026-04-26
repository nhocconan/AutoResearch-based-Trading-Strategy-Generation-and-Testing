#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1dTrend_VolumeSpike
Hypothesis: Ichimoku TK cross on 6h with 1d trend filter (price > 1d EMA50) and volume confirmation. Works in bull/bear via 1d trend filter. TK cross provides timely entries, cloud acts as dynamic support/resistance. Volume spike confirms breakout strength. Targets 50-150 total trades over 4 years via multiple filters. ATR-based stoploss for risk control.
"""

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
    
    # Load 1d data ONCE before loop for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    period_senkou_b = 52
    high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((high_senkou_b + low_senkou_b) / 2)
    
    # Chikou Span (Lagging Span): Close plotted 26 periods behind
    # Not used for signals as it requires future data
    
    # Volume spike: volume > 1.5x 20-period median volume
    volume_series = pd.Series(volume)
    vol_median_20 = volume_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (1.5 * vol_median_20)
    
    # ATR(14) for stoploss
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 52 for Senkou B, 26 for Kijun, 9 for Tenkan, 20 for volume median, 14 for ATR
    start_idx = max(52, 26, 9, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(tenkan[i]) or
            np.isnan(kijun[i]) or
            np.isnan(senkou_a[i]) or
            np.isnan(senkou_b[i]) or
            np.isnan(vol_median_20[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        senkou_a_val = senkou_a[i]
        senkou_b_val = senkou_b[i]
        ema_50_val = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr_14[i]
        size = fixed_size
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        if position == 0:
            # Flat - look for entry
            # TK cross bullish: Tenkan crosses above Kijun
            tk_bullish = (tenkan_val > kijun_val) and (tenkan[i-1] <= kijun[i-1])
            # TK cross bearish: Tenkan crosses below Kijun
            tk_bearish = (tenkan_val < kijun_val) and (tenkan[i-1] >= kijun[i-1])
            
            # Long: TK bullish + price above cloud + volume spike + uptrend (close > 1d EMA50)
            long_entry = tk_bullish and (close_val > cloud_top) and vol_spike and (close_val > ema_50_val)
            # Short: TK bearish + price below cloud + volume spike + downtrend (close < 1d EMA50)
            short_entry = tk_bearish and (close_val < cloud_bottom) and vol_spike and (close_val < ema_50_val)
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on TK cross bearish, price below cloud, ATR stoploss, or trend reversal
            tk_bearish = (tenkan_val < kijun_val) and (tenkan[i-1] >= kijun[i-1])
            stop_price = entry_price - 2.0 * atr_val
            if tk_bearish or (close_val < cloud_bottom) or (close_val < stop_price) or (close_val < ema_50_val):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on TK cross bullish, price above cloud, ATR stoploss, or trend reversal
            tk_bullish = (tenkan_val > kijun_val) and (tenkan[i-1] <= kijun[i-1])
            stop_price = entry_price + 2.0 * atr_val
            if tk_bullish or (close_val > cloud_top) or (close_val > stop_price) or (close_val > ema_50_val):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0