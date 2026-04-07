#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Ichimoku Cloud with 1-day volume confirmation and 1-week EMA trend filter
# Long when price above Kumo (cloud) + Tenkan > Kijun + volume > 1.5x 20-period average + weekly EMA20 > EMA50
# Short when price below Kumo + Tenkan < Kijun + volume > 1.5x 20-period average + weekly EMA20 < EMA50
# Exit when price crosses Tenkan-Kijun midpoint (TK cross) in opposite direction
# Stoploss at 2.5 * ATR(20)
# Position size: 0.25 (25% of capital)
# Uses Ichimoku for trend/filter, volume for confirmation, weekly EMA for regime
# Target: 80-200 total trades over 4 years (20-50/year)

name = "6h_ichimoku_1d_vol_1w_ema_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for volume confirmation and Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # 1-week data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Ichimoku components (9, 26, 52 periods)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): close shifted 26 periods behind (not used for signals)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # 1-day volume average (20-period)
    volume_1d = df_1d['volume'].values
    volume_1d_s = pd.Series(volume_1d)
    volume_ma = volume_1d_s.rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma)
    
    # 1-week EMA trend filter (20 and 50)
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # ATR(20) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(52, n):
        # Skip if required data not available
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Kumo (cloud) boundaries - future cloud values
        senkou_top = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        senkou_bottom = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # TK Cross (Tenkan-Kijun midpoint)
        tk_mid = (tenkan_aligned[i] + kijun_aligned[i]) / 2
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below TK midpoint
            elif close[i] < tk_mid:
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
            # Exit: price crosses above TK midpoint
            elif close[i] > tk_mid:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Ichimoku signals with volume confirmation and weekly EMA filter
            # Volume filter: volume > 1.5x 20-period average
            volume_filter = volume[i] > 1.5 * volume_ma_aligned[i]
            # Trend filter: weekly EMA20 > EMA50 for long, EMA20 < EMA50 for short
            
            # Long: price above cloud + Tenkan > Kijun + volume filter + weekly EMA20 > EMA50
            if (close[i] > senkou_top and close[i] > senkou_bottom and  # price above cloud
                tenkan_aligned[i] > kijun_aligned[i] and  # Tenkan > Kijun
                volume_filter and 
                ema20_1w_aligned[i] > ema50_1w_aligned[i]):  # weekly uptrend
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price below cloud + Tenkan < Kijun + volume filter + weekly EMA20 < EMA50
            elif (close[i] < senkou_top and close[i] < senkou_bottom and  # price below cloud
                  tenkan_aligned[i] < kijun_aligned[i] and  # Tenkan < Kijun
                  volume_filter and 
                  ema20_1w_aligned[i] < ema50_1w_aligned[i]):  # weekly downtrend
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals