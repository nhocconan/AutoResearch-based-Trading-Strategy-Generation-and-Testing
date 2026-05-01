#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation.
# Long when price breaks above Kumo (cloud) AND Tenkan > Kijun (bullish TK cross) AND price > 1d EMA50 (uptrend) AND volume > 1.5x 20-period volume median.
# Short when price breaks below Kumo (cloud) AND Tenkan < Kijun (bearish TK cross) AND price < 1d EMA50 (downtrend) AND volume > 1.5x 20-period volume median.
# Exit when price re-enters the Kumo (cloud) to reduce whipsaw.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year on 6h.
# Works in bull (buy cloud breakouts with trend) and bear (sell cloud breakdowns with trend).

name = "6h_Ichimoku_Cloud_Breakout_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (high_senkou_b + low_senkou_b) / 2
    
    # Chikou Span (Lagging Span): not used for breakout signals
    
    # The Kumo (cloud) is between Senkou Span A and Senkou Span B
    # Upper cloud boundary: max(Senkou A, Senkou B)
    # Lower cloud boundary: min(Senkou A, Senkou B)
    upper_cloud = np.maximum(senkou_a, senkou_b)
    lower_cloud = np.minimum(senkou_a, senkou_b)
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for Ichimoku (max period 52) and EMA50
    start_idx = 60
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i]) or
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 1d EMA50 direction
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        # TK cross: Tenkan > Kijun (bullish) or Tenkan < Kijun (bearish)
        tk_bullish = tenkan[i] > kijun[i]
        tk_bearish = tenkan[i] < kijun[i]
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        # Kumo breakout conditions
        breakout_above_cloud = curr_close > upper_cloud[i]   # break above cloud
        breakout_below_cloud = curr_close < lower_cloud[i]   # break below cloud
        reenter_cloud = (curr_close >= lower_cloud[i]) and (curr_close <= upper_cloud[i])  # price back in cloud
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above cloud AND bullish TK cross AND uptrend AND volume confirmation
            if breakout_above_cloud and tk_bullish and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below cloud AND bearish TK cross AND downtrend AND volume confirmation
            elif breakout_below_cloud and tk_bearish and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit when price re-enters the cloud (cloud acts as dynamic support/resistance)
            if reenter_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price re-enters the cloud
            if reenter_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals