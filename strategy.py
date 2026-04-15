#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation
# Long when price breaks above 6h Ichimoku Cloud (Senkou Span A/B) + 1d close > 1d EMA50 (uptrend) + volume > 1.3x 20-period avg
# Short when price breaks below 6h Ichimoku Cloud + 1d close < 1d EMA50 (downtrend) + volume > 1.3x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (12-30/year).
# Ichimoku Cloud provides dynamic support/resistance. 1d EMA50 filter ensures we trade with the higher timeframe trend.
# Works in bull markets (buying breakouts in uptrend) and bear markets (selling breakdowns in downtrend).

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
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d Indicator: EMA50 (trend filter) ===
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === 6h Ichimoku Cloud Components ===
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    highest_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    lowest_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (highest_high_tenkan + lowest_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    highest_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    lowest_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (highest_high_kijun + lowest_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, plotted 26 periods ahead
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, plotted 26 periods ahead
    period_senkou_b = 52
    highest_high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    lowest_low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = (highest_high_senkou_b + lowest_low_senkou_b) / 2
    
    # The Cloud (Kumo) is between Senkou Span A and Senkou Span B
    # Upper Cloud = max(Senkou Span A, Senkou Span B)
    # Lower Cloud = min(Senkou Span A, Senkou Span B)
    upper_cloud = np.maximum(senkou_span_a, senkou_span_b)
    lower_cloud = np.minimum(senkou_span_a, senkou_span_b)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    # Need max of: tenkan(9), kijun(26), senkou_b(52) + 26 displacement = 78, plus volume(20)
    warmup = 78 + 20
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.3)
        
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or
            np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned 1d close for trend filter
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)[i]
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above upper cloud (close > upper_cloud)
        # 2. 1d close > 1d EMA50 (uptrend)
        # 3. Volume confirmation
        if (close[i] > upper_cloud[i]) and \
           (close_1d_aligned > ema50_1d_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below lower cloud (close < lower_cloud)
        # 2. 1d close < 1d EMA50 (downtrend)
        # 3. Volume confirmation
        elif (close[i] < lower_cloud[i]) and \
             (close_1d_aligned < ema50_1d_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_IchimokuCloud_1dEMA50_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0