#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Ichimoku Cloud strategy with 1-day trend filter
# Long when Tenkan-sen > Kijun-sen (bullish TK cross), price above Kumo (cloud), and price > 1d EMA(50)
# Short when Tenkan-sen < Kijun-sen (bearish TK cross), price below Kumo, and price < 1d EMA(50)
# Exit when TK cross reverses or price crosses 1d EMA
# Stoploss at 2.5 * ATR(20)
# Position size: 0.25 (25% of capital)
# Uses Ichimoku for trend/momentum and 1d EMA to avoid counter-trend trades
# Target: 80-180 trades over 4 years (20-45/year)

name = "6h_ichimoku_1d_ema_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Ichimoku components (9, 26, 52 periods)
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
    
    # Kumo (Cloud) boundaries: Senkou Span A and B shifted forward 26 periods
    # For cloud at current point, we need values from 26 periods ago
    senkou_span_a_lag = np.roll(senkou_span_a, 26)
    senkou_span_b_lag = np.roll(senkou_span_b, 26)
    # First 26 values are invalid due to look-ahead, set to NaN
    senkou_span_a_lag[:26] = np.nan
    senkou_span_b_lag[:26] = np.nan
    
    # Cloud top and bottom
    kumo_top = np.maximum(senkou_span_a_lag, senkou_span_b_lag)
    kumo_bottom = np.minimum(senkou_span_a_lag, senkou_span_b_lag)
    
    # ATR(20) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(52, n):  # Need enough data for Ichimoku
        # Skip if required data not available
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(tenkan_sen[i]) or 
            np.isnan(kijun_sen[i]) or np.isnan(kumo_top[i]) or 
            np.isnan(kumo_bottom[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: TK cross turns bearish or price below EMA (trend change)
            elif tenkan_sen[i] < kijun_sen[i] or close[i] < ema_1d_aligned[i]:
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
            # Exit: TK cross turns bullish or price above EMA (trend change)
            elif tenkan_sen[i] > kijun_sen[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with TK cross and trend alignment
            # Bullish TK cross: Tenkan crosses above Kijun
            bullish_tk = tenkan_sen[i] > kijun_sen[i] and tenkan_sen[i-1] <= kijun_sen[i-1]
            # Bearish TK cross: Tenkan crosses below Kijun
            bearish_tk = tenkan_sen[i] < kijun_sen[i] and tenkan_sen[i-1] >= kijun_sen[i-1]
            
            # Price above/below cloud
            price_above_kumo = close[i] > kumo_top[i]
            price_below_kumo = close[i] < kumo_bottom[i]
            
            # Long: bullish TK cross, price above cloud, price above 1d EMA (bullish trend)
            if bullish_tk and price_above_kumo and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: bearish TK cross, price below cloud, price below 1d EMA (bearish trend)
            elif bearish_tk and price_below_kumo and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals