#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation
# Long when price breaks above 6h Ichimoku cloud (Tenkan/Kijun from 6h) + 1d price > Senkou Span A/B + volume > 1.5x 20-period avg
# Short when price breaks below 6h Ichimoku cloud + 1d price < Senkou Span A/B + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (12-30/year).
# Ichimoku cloud acts as dynamic support/resistance. 1d filter ensures alignment with higher timeframe trend.
# Works in bull markets (cloud acts as support in uptrends) and bear markets (cloud acts as resistance in downtrends).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicator: Ichimoku Cloud (Senkou Span A/B) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    high_10 = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_10 = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_10 + low_10) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    high_20 = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_20 = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_20 + low_20) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Shift forward 26 periods (we'll align later with align_htf_to_ltf which handles completed bar timing)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    high_40 = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_40 = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((high_40 + low_40) / 2)
    
    # Align 1d Ichimoku components to 6h timeframe (completed bar timing)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a, additional_delay_bars=26)  # Extra delay for leading span
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b, additional_delay_bars=26)  # Extra delay for leading span
    
    # === 6h Indicator: Ichimoku Cloud (Tenkan/Kijun) ===
    period_tenkan_6h = 9
    period_kijun_6h = 26
    
    high_9 = pd.Series(high).rolling(window=period_tenkan_6h, min_periods=period_tenkan_6h).max().values
    low_9 = pd.Series(low).rolling(window=period_tenkan_6h, min_periods=period_tenkan_6h).min().values
    tenkan_6h = (high_9 + low_9) / 2
    
    high_26 = pd.Series(high).rolling(window=period_kijun_6h, min_periods=period_kijun_6h).max().values
    low_26 = pd.Series(low).rolling(window=period_kijun_6h, min_periods=period_kijun_6h).min().values
    kijun_6h = (high_26 + low_26) / 2
    
    # Senkou Span A (6h): (Tenkan_6h + Kijun_6h)/2 shifted 26 periods ahead
    senkou_a_6h = ((tenkan_6h + kijun_6h) / 2)
    # Senkou Span B (6h): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b_6h = 52
    high_52 = pd.Series(high).rolling(window=period_senkou_b_6h, min_periods=period_senkou_b_6h).max().values
    low_52 = pd.Series(low).rolling(window=period_senkou_b_6h, min_periods=period_senkou_b_6h).min().values
    senkou_b_6h = ((high_52 + low_52) / 2)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(52, 26) + 20  # Ichimoku(52) + volume(20)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or
            np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Calculate 6h cloud boundaries (top and bottom of cloud)
        cloud_top_6h = np.maximum(senkou_a_6h[i], senkou_b_6h[i])
        cloud_bottom_6h = np.minimum(senkou_a_6h[i], senkou_b_6h[i])
        
        # Calculate 1d cloud boundaries (top and bottom of cloud)
        cloud_top_1d = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom_1d = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 6h Ichimoku cloud
        # 2. 1d price above 1d Ichomoku cloud (trend filter)
        # 3. Volume confirmation
        if (close[i] > cloud_top_6h) and \
           (close[i] > cloud_top_1d) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 6h Ichimoku cloud
        # 2. 1d price below 1d Ichimoku cloud (trend filter)
        # 3. Volume confirmation
        elif (close[i] < cloud_bottom_6h) and \
             (close[i] < cloud_bottom_1d) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_Filter_v1"
timeframe = "6h"
leverage = 1.0