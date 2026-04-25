#!/usr/bin/env python3
"""
6h Ichimoku Cloud Breakout with Weekly Kumo Twist Filter
Hypothesis: Ichimoku captures trend, momentum, and support/resistance. TK cross signals momentum shifts,
while price > cloud confirms uptrend and price < cloud confirms downtrend. Weekly Kumo Twist (Senkou
Span A/B cross) indicates major trend regime change, filtering counter-trend signals. Works in bull
(by catching breakouts above cloud with bullish TK cross) and bear (breakouts below cloud with bearish
TK cross) regimes. Targets 12-37 trades/year by requiring TK cross, cloud alignment, and weekly filter.
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
    
    # Load daily data ONCE before loop for Ichimoku and weekly filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Ichimoku components (9, 26, 52 periods)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Current Kumo (cloud) top/bottom: Senkou Span shifted back 26 periods to align with current price
    # We need the cloud values that were plotted 26 periods ago for today's price
    cloud_top = np.roll(senkou_a, 26)  # Senkou A from 26 periods ago
    cloud_bottom = np.roll(senkou_b, 26)  # Senkou B from 26 periods ago
    # For first 26 periods, cloud data isn't available -> will be handled by min_periods/checks
    
    # TK Cross: Tenkan crossing above/below Kijun
    tk_cross_above = (tenkan > kijun) & (np.roll(tenkan, 1) <= np.roll(kijun, 1))
    tk_cross_below = (tenkan < kijun) & (np.roll(tenkan, 1) >= np.roll(kijun, 1))
    
    # Weekly Kumo Twist: Senkou A/B cross on weekly timeframe
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    # Weekly Ichimoku for Kumo Twist
    wh_high = df_1w['high'].values
    wh_low = df_1w['low'].values
    # Weekly Tenkan/Kijun
    w_period9_high = pd.Series(wh_high).rolling(window=9, min_periods=9).max().values
    w_period9_low = pd.Series(wh_low).rolling(window=9, min_periods=9).min().values
    w_tenkan = (w_period9_high + w_period9_low) / 2
    w_period26_high = pd.Series(wh_high).rolling(window=26, min_periods=26).max().values
    w_period26_low = pd.Series(wh_low).rolling(window=26, min_periods=26).min().values
    w_kijun = (w_period26_high + w_period26_low) / 2
    # Weekly Senkou Span A/B
    w_senkou_a = (w_tenkan + w_kijun) / 2
    w_period52_high = pd.Series(wh_high).rolling(window=52, min_periods=52).max().values
    w_period52_low = pd.Series(wh_low).rolling(window=52, min_periods=52).min().values
    w_senkou_b = (w_period52_high + w_period52_low) / 2
    # Kumo Twist: Weekly Senkou A crossing Senkou B
    wkumo_twist_above = (w_senkou_a > w_senkou_b) & (np.roll(w_senkou_a, 1) <= np.roll(w_senkou_b, 1))
    wkumo_twist_below = (w_senkou_a < w_senkou_b) & (np.roll(w_senkou_a, 1) >= np.roll(w_senkou_b, 1))
    # Current weekly Kumo twist state (bullish if Senkou A > Senkou B)
    wkumo_bullish = w_senkou_a > w_senkou_b
    wkumo_bearish = w_senkou_a < w_senkou_b
    
    # Align all 1d Ichimoku to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    cloud_top_6h = align_htf_to_ltf(prices, df_1d, cloud_top)
    cloud_bottom_6h = align_htf_to_ltf(prices, df_1d, cloud_bottom)
    tk_cross_above_6h = align_htf_to_ltf(prices, df_1d, tk_cross_above.astype(float))
    tk_cross_below_6h = align_htf_to_ltf(prices, df_1d, tk_cross_below.astype(float))
    
    # Align weekly Kumo twist to 6h
    wkumo_twist_above_6h = align_htf_to_ltf(prices, df_1w, wkumo_twist_above.astype(float), additional_delay_bars=0)
    wkumo_twist_below_6h = align_htf_to_ltf(prices, df_1w, wkumo_twist_below.astype(float), additional_delay_bars=0)
    wkumo_bullish_6h = align_htf_to_ltf(prices, df_1w, wkumo_bullish.astype(float), additional_delay_bars=0)
    wkumo_bearish_6h = align_htf_to_ltf(prices, df_1w, wkumo_bearish.astype(float), additional_delay_bars=0)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Ichimoku calculations
    start_idx = max(52, 26)  # Ichimoku needs 52 for Senkou B, plus alignment
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(cloud_top_6h[i]) or np.isnan(cloud_bottom_6h[i]) or
            np.isnan(tk_cross_above_6h[i]) or np.isnan(tk_cross_below_6h[i]) or
            np.isnan(wkumo_bullish_6h[i]) or np.isnan(wkumo_bearish_6h[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Price relative to cloud
        price_above_cloud = curr_close > max(cloud_top_6h[i], cloud_bottom_6h[i])
        price_below_cloud = curr_close < min(cloud_top_6h[i], cloud_bottom_6h[i])
        price_in_cloud = not (price_above_cloud or price_below_cloud)
        
        # Trend bias from weekly Kumo twist
        bullish_regime = wkumo_bullish_6h[i] == 1.0
        bearish_regime = wkumo_bearish_6h[i] == 1.0
        
        if position == 0:
            # Look for entry signals
            # Long: TK cross bullish + price above cloud + bullish weekly regime + volume spike
            long_entry = (tk_cross_above_6h[i] == 1.0 and 
                         price_above_cloud and 
                         bullish_regime and 
                         vol_spike)
            # Short: TK cross bearish + price below cloud + bearish weekly regime + volume spike
            short_entry = (tk_cross_below_6h[i] == 1.0 and 
                          price_below_cloud and 
                          bearish_regime and 
                          vol_spike)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: TK cross bearish OR price falls into/below cloud OR loss of bullish regime
            if (tk_cross_below_6h[i] == 1.0 or 
                price_in_cloud or price_below_cloud or 
                bearish_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: TK cross bullish OR price rises into/above cloud OR loss of bearish regime
            if (tk_cross_above_6h[i] == 1.0 or 
                price_in_cloud or price_above_cloud or 
                bullish_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_WeeklyKumoTwist_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0