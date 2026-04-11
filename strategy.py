#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ichimoku_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 periods for Ichimoku
        return signals
    
    # Calculate Ichimoku components on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Chikou Span (Lagging Span): current close plotted 26 periods back
    chikou = np.roll(close_1d, 26)
    
    # Shift all Ichimoku components by 1 to use only completed daily bars
    tenkan = np.roll(tenkan, 1)
    kijun = np.roll(kijun, 1)
    senkou_a = np.roll(senkou_a, 1)
    senkou_b = np.roll(senkou_b, 1)
    chikou = np.roll(chikou, 1)
    tenkan[0] = np.nan
    kijun[0] = np.nan
    senkou_a[0] = np.nan
    senkou_b[0] = np.nan
    chikou[0] = np.nan
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    chikou_aligned = align_htf_to_ltf(prices, df_1d, chikou)
    
    # Calculate 6h ADX for trend strength filter
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), high[1:] - high[:-1], 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), low[:-1] - low[1:], 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan
    
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if any required data is invalid
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(chikou_aligned[i]) or np.isnan(adx[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.3 * vol_ma
        
        # Determine cloud color and position
        # If Senkou A > Senkou B, bullish cloud (green)
        # If Senkou A < Senkou B, bearish cloud (red)
        bullish_cloud = senkou_a_aligned[i] > senkou_b_aligned[i]
        bearish_cloud = senkou_a_aligned[i] < senkou_b_aligned[i]
        
        # Price above/below cloud
        above_cloud = price_close > max(senkou_a_aligned[i], senkou_b_aligned[i])
        below_cloud = price_close < min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Long conditions: TK cross bullish, price above cloud, ADX > 20, volume confirmed
        tk_bullish = tenkan_aligned[i] > kijun_aligned[i]
        long_signal = volume_confirmed and tk_bullish and above_cloud and (adx[i] > 20)
        
        # Short conditions: TK cross bearish, price below cloud, ADX > 20, volume confirmed
        tk_bearish = tenkan_aligned[i] < kijun_aligned[i]
        short_signal = volume_confirmed and tk_bearish and below_cloud and (adx[i] > 20)
        
        # Exit when TK cross reverses or price enters cloud
        exit_long = position == 1 and (not tk_bullish or not above_cloud)
        exit_short = position == -1 and (not tk_bearish or not below_cloud)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
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

# Hypothesis: Ichimoku system on daily timeframe with 6h ADX filter and volume confirmation.
# Uses Ichimoku cloud (Senkou Span A/B) to identify trend direction and support/resistance.
# Tenkan/Kijun cross provides entry signals, with price position relative to cloud
# filtering for trend strength. ADX > 20 ensures we only trade in trending markets.
# Volume confirmation ensures participation. Works in both bull and bear markets by
# following the Ichimoku trend signals. Target: 50-150 total trades over 4 years
# (12-37/year) to minimize fee drag on 6h timeframe. Ichimoku is a comprehensive
# system that works well in crypto markets for identifying trends and reversals.