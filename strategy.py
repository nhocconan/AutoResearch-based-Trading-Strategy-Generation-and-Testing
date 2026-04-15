#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation
# Long when: price > Kumo cloud (Senkou Span A/B), Tenkan > Kijun, 1d EMA50 uptrend, volume > 1.5x 20-period avg
# Short when: price < Kumo cloud, Tenkan < Kijun, 1d EMA50 downtrend, volume > 1.5x 20-period avg
# Uses Kumo cloud as dynamic support/resistance, Tenkan/Kijun for momentum, 1d EMA50 for higher-timeframe trend.
# Volume filter reduces false breakouts. Designed for 6h timeframe targeting 12-30 trades/year.
# Works in bull markets (trend following via cloud) and bear markets (short signals below cloud).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicator: EMA50 ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 6h Ichimoku Components (9, 26, 52 periods) ===
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, plotted 26 periods ahead
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52 + low_52) / 2)
    
    # Kumo cloud boundaries (shifted 26 periods ahead for plotting, but we use current values)
    # For current cloud, we use Senkou A/B values that were calculated 26 periods ago
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    # First 26 values are invalid due to roll
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Upper cloud = max(Senkou A, Senkou B), Lower cloud = min(Senkou A, Senkou B)
    upper_cloud = np.where(senkou_a_shifted > senkou_b_shifted, senkou_a_shifted, senkou_b_shifted)
    lower_cloud = np.where(senkou_a_shifted < senkou_b_shifted, senkou_a_shifted, senkou_b_shifted)
    
    # Volume SMA for confirmation (20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(52, 26, 9, 50) + 5  # Ichimoku(52) + EMA50 + volume(20) + buffer
    
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
        # 1. Price above Kumo cloud (close > upper cloud)
        # 2. Tenkan > Kijun (bullish momentum)
        # 3. 1d EMA50 uptrend (close > EMA50)
        # 4. Volume confirmation
        if (close[i] > upper_cloud[i]) and \
           (tenkan[i] > kijun[i]) and \
           (close[i] > ema_50_1d_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price below Kumo cloud (close < lower cloud)
        # 2. Tenkan < Kijun (bearish momentum)
        # 3. 1d EMA50 downtrend (close < EMA50)
        # 4. Volume confirmation
        elif (close[i] < lower_cloud[i]) and \
             (tenkan[i] < kijun[i]) and \
             (close[i] < ema_50_1d_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Ichimoku_1dEMA50_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0