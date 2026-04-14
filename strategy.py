#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Ichimoku Cloud with 1w ADX Trend Filter
# Uses Ichimoku Cloud (tenkan/kijun/senkou) for trend-following entries
# 1w ADX (>25) filters for strong trending conditions to avoid whipsaws
# Price above/below cloud with TK cross provides entry/exit signals
# Works in bull/bear by only trading with strong trends (ADX filter)
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1w data ONCE before loop for ADX filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w ADX (14) for trend strength filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI and DX
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_1w = adx
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Ichimoku Cloud calculations (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    nine_period_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    nine_period_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (nine_period_high + nine_period_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    twenty_six_period_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    twenty_six_period_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (twenty_six_period_high + twenty_six_period_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    fifty_two_period_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    fifty_two_period_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (fifty_two_period_high + fifty_two_period_low) / 2
    
    # Shift Senkou Spans forward by 26 periods
    senkou_a_shifted = np.concatenate([np.full(26, np.nan), senkou_a[:-26]])
    senkou_b_shifted = np.concatenate([np.full(26, np.nan), senkou_b[:-26]])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 52  # for Ichimoku calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a_shifted[i]) or np.isnan(senkou_b_shifted[i]) or
            np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        adx_value = adx_1w_aligned[i]
        
        # Trend filter: only trade when ADX > 25 (strong trend)
        strong_trend = adx_value > 25
        
        # Cloud boundaries
        upper_cloud = np.maximum(senkou_a_shifted[i], senkou_b_shifted[i])
        lower_cloud = np.minimum(senkou_a_shifted[i], senkou_b_shifted[i])
        
        if position == 0:
            # Long: price above cloud AND TK cross bullish (tenkan > kijun)
            if price > upper_cloud and tenkan[i] > kijun[i] and strong_trend:
                position = 1
                signals[i] = position_size
            # Short: price below cloud AND TK cross bearish (tenkan < kijun)
            elif price < lower_cloud and tenkan[i] < kijun[i] and strong_trend:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls below cloud OR TK cross turns bearish
            if price < lower_cloud or tenkan[i] < kijun[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price rises above cloud OR TK cross turns bullish
            if price > upper_cloud or tenkan[i] > kijun[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_Ichimoku_Cloud_1wADX_Trend"
timeframe = "1d"
leverage = 1.0