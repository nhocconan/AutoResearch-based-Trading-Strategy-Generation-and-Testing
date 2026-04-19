#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with weekly trend filter and volume confirmation.
# Ichimoku provides dynamic support/resistance via Kumo (cloud). TK cross (Tenkan/Kijun) signals momentum shifts.
# Weekly trend filter (price vs weekly Kumo) ensures alignment with higher timeframe trend.
# Volume confirmation filters false breakouts. Works in bull/bear markets: cloud acts as dynamic S/R in trends,
# and TK cross captures momentum shifts. Target: 12-30 trades/year per symbol.
name = "6h_Ichimoku_TK_Cross_WeeklyTrend_Volume"
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
    
    # Ichimoku parameters (6h)
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    displacement = 26
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                  pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                 pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b = ((pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                      pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2).shift(displacement)
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) == 0:
        return np.zeros(n)
    
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    # Weekly Ichimoku components (same parameters)
    weekly_tenkan = (pd.Series(weekly_high).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                     pd.Series(weekly_low).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    weekly_kijun = (pd.Series(weekly_high).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                    pd.Series(weekly_low).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    weekly_senkou_a = ((weekly_tenkan + weekly_kijun) / 2).shift(displacement)
    weekly_senkou_b = ((pd.Series(weekly_high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                        pd.Series(weekly_low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2).shift(displacement)
    
    # Align weekly Ichimoku to 6h
    weekly_senkou_a_aligned = align_htf_to_ltf(prices, df_weekly, weekly_senkou_a.values)
    weekly_senkou_b_aligned = align_htf_to_ltf(prices, df_weekly, weekly_senkou_b.values)
    
    # Weekly Kumo (cloud) boundaries
    weekly_kumo_top = np.maximum(weekly_senkou_a_aligned, weekly_senkou_b_aligned)
    weekly_kumo_bottom = np.minimum(weekly_senkou_a_aligned, weekly_senkou_b_aligned)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(tenkan_period, kijun_period, senkou_span_b_period, displacement, 20) + displacement
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or np.isnan(senkou_span_a[i]) or 
            np.isnan(senkou_span_b[i]) or np.isnan(weekly_kumo_top[i]) or np.isnan(weekly_kumo_bottom[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        tk_cross = tenkan_sen[i] - kijun_sen[i]
        prev_tk_cross = tenkan_sen[i-1] - kijun_sen[i-1] if i > 0 else 0
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        weekly_top = weekly_kumo_top[i]
        weekly_bottom = weekly_kumo_bottom[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        # Price relative to weekly cloud
        price_above_weekly_kumo = price > weekly_top
        price_below_weekly_kumo = price < weekly_bottom
        
        if position == 0:
            # Long: TK cross turns positive (bullish momentum) + price above weekly cloud + volume
            if tk_cross > 0 and prev_tk_cross <= 0 and price_above_weekly_kumo and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: TK cross turns negative (bearish momentum) + price below weekly cloud + volume
            elif tk_cross < 0 and prev_tk_cross >= 0 and price_below_weekly_kumo and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when TK cross turns negative (momentum shift) or price enters weekly cloud
            if tk_cross < 0 or (price >= weekly_bottom and price <= weekly_top):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when TK cross turns positive (momentum shift) or price enters weekly cloud
            if tk_cross > 0 or (price >= weekly_bottom and price <= weekly_top):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals