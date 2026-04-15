#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation
# Long when price breaks above Ichimoku cloud (Senkou Span A/B) + 1d EMA50 > EMA200 (bullish trend) + volume > 1.5x 20-period avg
# Short when price breaks below Ichimoku cloud + 1d EMA50 < EMA200 (bearish trend) + volume > 1.5x 20-period avg
# Ichimoku provides dynamic support/resistance via cloud. EMA50/200 filter ensures we trade with higher timeframe trend.
# Works in bull markets (cloud acts as support in uptrends) and bear markets (cloud acts as resistance in downtrends).
# Designed for low trade frequency (12-30/year) with discrete sizing (0.25) to minimize fee churn.

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
    
    # === 1d Indicators: EMA50 and EMA200 (trend filter) ===
    close_1d = df_1d['close'].values
    
    # EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # EMA200
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # === 6h Indicator: Ichimoku Cloud ===
    # Conversion Line (Tenkan-sen): (9-period high + 9-period low) / 2
    period_9 = 9
    high_9 = pd.Series(high).rolling(window=period_9, min_periods=period_9).max().values
    low_9 = pd.Series(low).rolling(window=period_9, min_periods=period_9).min().values
    tenkan_sen = (high_9 + low_9) / 2
    
    # Base Line (Kijun-sen): (26-period high + 26-period low) / 2
    period_26 = 26
    high_26 = pd.Series(high).rolling(window=period_26, min_periods=period_26).max().values
    low_26 = pd.Series(low).rolling(window=period_26, min_periods=period_26).min().values
    kijun_sen = (high_26 + low_26) / 2
    
    # Leading Span A (Senkou Span A): (Conversion Line + Base Line) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Leading Span B (Senkou Span B): (52-period high + 52-period low) / 2
    period_52 = 52
    high_52 = pd.Series(high).rolling(window=period_52, min_periods=period_52).max().values
    low_52 = pd.Series(low).rolling(window=period_52, min_periods=period_52).min().values
    senkou_span_b = (high_52 + low_52) / 2
    
    # The cloud is between Senkou Span A and B
    # We need to shift them forward by 26 periods (but alignment handles timing)
    # For breakout, we compare current price to current cloud (unshifted)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(52, 200) + 20  # Ichimoku(52) + EMA200(200) + volume(20)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (upper and lower band)
        upper_cloud = np.maximum(senkou_span_a[i], senkou_span_b[i])
        lower_cloud = np.minimum(senkou_span_a[i], senkou_span_b[i])
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Ichimoku cloud (close > upper cloud)
        # 2. Bullish 1d trend (EMA50 > EMA200)
        # 3. Volume confirmation
        if (close[i] > upper_cloud) and \
           (ema50_1d_aligned[i] > ema200_1d_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Ichimoku cloud (close < lower cloud)
        # 2. Bearish 1d trend (EMA50 < EMA200)
        # 3. Volume confirmation
        elif (close[i] < lower_cloud) and \
             (ema50_1d_aligned[i] < ema200_1d_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0