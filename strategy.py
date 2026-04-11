#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_ichimoku_trend_follow"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load daily and weekly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 52 or len(df_1w) < 26:
        return signals
    
    # Calculate Ichimoku components on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou_span = close_1d
    
    # Shift Senkou spans for plotting (but we'll use them without shift for cloud calculation)
    # For cloud, we need Senkou A and B without forward shift to represent current cloud
    # The cloud is formed by Senkou A and B plotted 26 periods ahead
    # So current cloud at time t is Senkou A[t-26] and Senkou B[t-26]
    senkou_a_shifted = np.roll(senkou_span_a, 26)
    senkou_b_shifted = np.roll(senkou_span_b, 26)
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_shifted)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_shifted)
    chikou_span_aligned = align_htf_to_ltf(prices, df_1d, chikou_span)
    
    # Weekly trend filter: price above/below weekly cloud
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Ichimoku (simplified: just need cloud for trend)
    period9_high_w = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    period9_low_w = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_sen_w = (period9_high_w + period9_low_w) / 2
    
    period26_high_w = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    period26_low_w = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_sen_w = (period26_high_w + period26_low_w) / 2
    
    senkou_span_a_w = ((tenkan_sen_w + kijun_sen_w) / 2)
    period52_high_w = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    period52_low_w = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_span_b_w = ((period52_high_w + period52_low_w) / 2)
    
    senkou_a_w_shifted = np.roll(senkou_span_a_w, 26)
    senkou_b_w_shifted = np.roll(senkou_span_b_w, 26)
    senkou_a_w_shifted[:26] = np.nan
    senkou_b_w_shifted[:26] = np.nan
    
    # Align weekly Ichimoku to 6h timeframe
    senkou_a_w_aligned = align_htf_to_ltf(prices, df_1w, senkou_a_w_shifted)
    senkou_b_w_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_w_shifted)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(senkou_a_w_aligned[i]) or np.isnan(senkou_b_w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        tenkan = tenkan_sen_aligned[i]
        kijun = kijun_sen_aligned[i]
        senkou_a = senkou_a_aligned[i]
        senkou_b = senkou_b_aligned[i]
        senkou_a_w = senkou_a_w_aligned[i]
        senkou_b_w = senkou_b_w_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Cloud boundaries (top and bottom of cloud)
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        
        # Weekly cloud boundaries for trend filter
        weekly_cloud_top = max(senkou_a_w, senkou_b_w)
        weekly_cloud_bottom = min(senkou_a_w, senkou_b_w)
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.3 * vol_ma
        
        # TK Cross signals
        tk_cross_up = tenkan > kijun and tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1]
        tk_cross_down = tenkan < kijun and tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1]
        
        # Price position relative to cloud
        price_above_cloud = price_close > cloud_top
        price_below_cloud = price_close < cloud_bottom
        price_in_cloud = not (price_above_cloud or price_below_cloud)
        
        # Weekly trend filter: price relative to weekly cloud
        weekly_uptrend = price_close > weekly_cloud_top
        weekly_downtrend = price_close < weekly_cloud_bottom
        
        # Entry signals
        long_signal = False
        short_signal = False
        
        # Long: TK cross up + price above cloud + weekly uptrend + volume
        if tk_cross_up and price_above_cloud and weekly_uptrend and volume_confirmed:
            long_signal = True
        
        # Short: TK cross down + price below cloud + weekly downtrend + volume
        if tk_cross_down and price_below_cloud and weekly_downtrend and volume_confirmed:
            short_signal = True
        
        # Exit conditions: TK cross in opposite direction or price returns to cloud
        exit_long = position == 1 and (tk_cross_down or price_close < (tenkan + kijun) / 2)
        exit_short = position == -1 and (tk_cross_up or price_close > (tenkan + kijun) / 2)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            entry_price = price_close
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            entry_price = price_close
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Ichimoku trend following on 6h with daily/weekly confluence.
# Enters long when TK line crosses up AND price is above daily cloud AND price is above weekly cloud (strong uptrend).
# Enters short when TK line crosses down AND price is below daily cloud AND price is below weekly cloud (strong downtrend).
# Uses weekly Ichimoku cloud as higher timeframe trend filter to avoid counter-trend trades.
# Volume confirmation ensures institutional participation.
# Works in both bull and bear markets by trading with the higher timeframe trend.
# Target: 50-150 total trades over 4 years (12-37/year) with strict entry conditions.