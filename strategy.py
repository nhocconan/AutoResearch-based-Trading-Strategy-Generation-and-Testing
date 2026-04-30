#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 12h trend filter and volume confirmation
# Ichimoku provides dynamic support/resistance via cloud (Senkou Span A/B) and momentum via TK cross
# 12h EMA50 filter ensures alignment with higher timeframe trend to avoid counter-trend trades
# Volume spike (1.8x 30-period average) confirms breakout validity
# Works in bull markets via breakouts above cloud and TK cross up, bear via breakdowns below cloud and TK cross down
# Discrete sizing 0.25 minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_Ichimoku_Cloud_Breakout_12hEMA50_Volume_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 12h data ONCE before loop (MTF Rule #1)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Ichimoku components (using 6h data)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (max_tenkan + min_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (max_kijun + min_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period_senkou_b = 52
    max_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = (max_senkou_b + min_senkou_b) / 2
    
    # Chikou Span (Lagging Span): not used for signals (plotted 26 periods behind)
    
    # Volume confirmation: volume > 1.8x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma_30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 52, 30)  # warmup for Ichimoku (52), volume MA (30), and 12h EMA (50)
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_tenkan = tenkan_sen[i]
        curr_kijun = kijun_sen[i]
        curr_senkou_a = senkou_span_a[i]
        curr_senkou_b = senkou_span_b[i]
        curr_volume_spike = volume_spike[i]
        
        # Determine cloud top and bottom
        cloud_top = max(curr_senkou_a, curr_senkou_b)
        cloud_bottom = min(curr_senkou_a, curr_senkou_b)
        
        # TK cross conditions
        tk_cross_up = curr_tenkan > curr_kijun
        tk_cross_down = curr_tenkan < curr_kijun
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: price above cloud AND TK cross up AND above 12h EMA50 (uptrend)
                if curr_close > cloud_top and tk_cross_up and curr_close > curr_ema_50_12h:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price below cloud AND TK cross down AND below 12h EMA50 (downtrend)
                elif curr_close < cloud_bottom and tk_cross_down and curr_close < curr_ema_50_12h:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price drops below cloud bottom OR TK cross down
            if curr_close < cloud_bottom or not tk_cross_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above cloud top OR TK cross up
            if curr_close > cloud_top or tk_cross_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals