#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation
# Long when: Tenkan > Kijun (bullish TK cross) + price above Kumo (cloud) + 1d EMA50 uptrend + volume > 1.5x 20-period avg
# Short when: Tenkan < Kijun (bearish TK cross) + price below Kumo (cloud) + 1d EMA50 downtrend + volume > 1.5x 20-period avg
# Uses Ichimoku from 6h timeframe for entry timing, 1d EMA50 for trend filter to avoid counter-trend trades
# Volume confirmation ensures breakouts have conviction
# Discrete position sizing (0.25) to control drawdown and minimize fee churn
# Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe

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
    
    # === 1d Indicator: EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 6h Ichimoku Components ===
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    high_9 = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_9 = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_9 + low_9) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    high_26 = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_26 = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_26 + low_26) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    high_52 = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_52 = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (high_52 + low_52) / 2.0
    
    # Kumo (Cloud) boundaries: Senkou Span A and B
    # Upper cloud = max(Senkou A, Senkou B)
    # Lower cloud = min(Senkou A, Senkou B)
    upper_cloud = np.maximum(senkou_a, senkou_b)
    lower_cloud = np.minimum(senkou_a, senkou_b)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(52, 50, 20) + 5  # Senkou B (52) + EMA50 (50) + volume (20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # === LONG CONDITIONS ===
        # 1. Bullish TK cross: Tenkan > Kijun
        # 2. Price above Kumo (cloud): close > upper cloud
        # 3. 1d EMA50 uptrend: close > EMA50
        # 4. Volume confirmation
        if (tenkan[i] > kijun[i]) and \
           (close[i] > upper_cloud[i]) and \
           (close[i] > ema_50_1d_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Bearish TK cross: Tenkan < Kijun
        # 2. Price below Kumo (cloud): close < lower cloud
        # 3. 1d EMA50 downtrend: close < EMA50
        # 4. Volume confirmation
        elif (tenkan[i] < kijun[i]) and \
             (close[i] < lower_cloud[i]) and \
             (close[i] < ema_50_1d_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Ichimoku_TK_Cross_Cloud_1dEMA50_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0