#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Ichimoku Cloud with 1-day trend filter and volume confirmation
# Long when Tenkan-sen > Kijun-sen (TK cross) + price above cloud (Senkou Span A/B) + 1-day ADX > 25 + volume > 1.5x 20-period average
# Short when Tenkan-sen < Kijun-sen + price below cloud + 1-day ADX > 25 + volume > 1.5x 20-period average
# Exit when TK cross reverses or price enters cloud
# Stoploss at 2.5 * ATR(22)
# Position size: 0.25 (25% of capital)
# Uses Ichimoku for trend/momentum, 1-day ADX for trend strength, volume for confirmation
# Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_ichimoku_1d_trend_filter_v3"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for ADX trend filter and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # 1-day ADX (14-period) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = t1[0]
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = np.diff(low_1d, prepend=low_1d[0]) * -1  # invert to positive
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / (tr_14 + 1e-10)
    minus_di = 100 * minus_dm_14 / (tr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 1-day volume average (20-period)
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # 6-hour ATR(22) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=22, adjust=False, min_periods=22).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(52, n):
        # Skip if required data not available
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(volume_ma_1d_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A/B shifted 26 periods ahead)
        # For current price, we use Senkou Span values calculated 26 periods ago
        span_a = senkou_span_a[i-26] if i >= 26+26 else senkou_span_a[0]
        span_b = senkou_span_b[i-26] if i >= 26+26 else senkou_span_b[0]
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        # TK cross
        tk_cross = tenkan_sen[i] - kijun_sen[i]
        tk_cross_prev = tenkan_sen[i-1] - kijun_sen[i-1]
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: TK cross turns negative or price enters cloud
            elif tk_cross <= 0 or (close[i] >= cloud_bottom and close[i] <= cloud_top):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: TK cross turns positive or price enters cloud
            elif tk_cross >= 0 or (close[i] >= cloud_bottom and close[i] <= cloud_top):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: TK cross with price outside cloud + ADX filter + volume confirmation
            # Volume filter: volume > 1.5x 20-period average
            volume_filter = volume[i] > 1.5 * volume_ma_1d_aligned[i]
            # Trend filter: 1-day ADX > 25
            trend_filter = adx_1d_aligned[i] > 25
            
            # Bullish TK cross: Tenkan crosses above Kijun
            bullish_cross = tk_cross > 0 and tk_cross_prev <= 0
            # Bearish TK cross: Tenkan crosses below Kijun
            bearish_cross = tk_cross < 0 and tk_cross_prev >= 0
            
            # Long: bullish TK cross + price above cloud + volume filter + trend filter
            if bullish_cross and close[i] > cloud_top and volume_filter and trend_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: bearish TK cross + price below cloud + volume filter + trend filter
            elif bearish_cross and close[i] < cloud_bottom and volume_filter and trend_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals