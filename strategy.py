#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation.
# Long when Tenkan-Sen crosses above Kijun-Sen AND price breaks above Kumo (cloud) top in bull trend (close > 1d EMA50) with volume spike.
# Short when Tenkan-Sen crosses below Kijun-Sen AND price breaks below Kumo (cloud) bottom in bear trend (close < 1d EMA50) with volume spike.
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for 50-150 total trades over 4 years (12-37/year).
# Ichimoku provides strong trend signals with built-in support/resistance (cloud). The 1d EMA50 filter ensures alignment with higher timeframe trend.
# Volume confirmation reduces false signals during low-participation breakouts.

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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 51:  # Need at least 50 for EMA + 1 for current
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku components on 6h timeframe
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    highest_high_9 = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    lowest_low_9 = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (highest_high_9 + lowest_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    highest_high_26 = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    lowest_low_26 = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (highest_high_26 + lowest_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    highest_high_52 = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    lowest_low_52 = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = (highest_high_52 + lowest_low_52) / 2
    
    # Kumo (cloud) top and bottom
    kumogun_top = np.maximum(senkou_span_a, senkou_span_b)
    kumogun_bottom = np.minimum(senkou_span_a, senkou_span_b)
    
    # Volume regime: current 6h volume > 2.0x 30-period MA (stricter for fewer trades)
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * vol_ma_30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(tenkan_sen[i]) or 
            np.isnan(kijun_sen[i]) or np.isnan(kumogun_top[i]) or 
            np.isnan(kumogun_bottom[i]) or np.isnan(vol_ma_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        tenkan = tenkan_sen[i]
        kijun = kijun_sen[i]
        kumo_top = kumogun_top[i]
        kumo_bottom = kumogun_bottom[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Ichimoku conditions
        tk_cross_up = tenkan > kijun  # Tenkan above Kijun
        tk_cross_down = tenkan < kijun  # Tenkan below Kijun
        price_above_kumo = close_val > kumo_top  # Price above cloud
        price_below_kumo = close_val < kumo_bottom  # Price below cloud
        
        # Entry logic
        if position == 0:
            if is_bull_trend and tk_cross_up and price_above_kumo and vol_spike:
                signals[i] = 0.25
                position = 1
            elif is_bear_trend and tk_cross_down and price_below_kumo and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Tenkan crosses below Kijun OR price falls below cloud bottom OR trend reversal
            if (tenkan < kijun) or (close_val < kumo_bottom) or (close_val < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Tenkan crosses above Kijun OR price rises above cloud top OR trend reversal
            if (tenkan > kijun) or (close_val > kumo_top) or (close_val > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals