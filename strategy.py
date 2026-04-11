#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud Breakout with Weekly Trend Filter
# - Tenkan-sen (9) and Kijun-sen (26) cross signals filtered by Kumo (cloud) color
# - Weekly trend filter: only trade long when weekly close > weekly SMA(50)
# - Only trade short when weekly close < weekly SMA(50)
# - Kumo (cloud) from Senkou Span A (26) and Senkou Span B (52) acts as support/resistance
# - Long when Tenkan crosses above Kijun AND price above Kumo AND weekly uptrend
# - Short when Tenkan crosses below Kijun AND price below Kumo AND weekly downtrend
# - Exit when Tenkan/Kijun cross reverses OR price crosses Kumo in opposite direction
# - Weekly filter ensures we only trade with the higher timeframe trend, reducing whipsaws
# - Target: 50-150 total trades over 4 years (12-37/year) to stay within 6h limits
# - Discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Works in both bull (trend following with weekly uptrend) and bear (trend following with weekly downtrend)

name = "6h_1w_ichimoku_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute weekly SMA(50) for trend filter
    weekly_close = df_1w['close'].values
    weekly_sma_50 = pd.Series(weekly_close).rolling(window=50, min_periods=50).mean().values
    weekly_sma_50_aligned = align_htf_to_ltf(prices, df_1w, weekly_sma_50)
    
    # Pre-compute Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low).rolling(window=9, min_periods=9).min()) / 2
    tenkan_sen = tenkan_sen.values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low).rolling(window=26, min_periods=26).min()) / 2
    kijun_sen = kijun_sen.values
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = ((pd.Series(high).rolling(window=52, min_periods=52).max() + 
                 pd.Series(low).rolling(window=52, min_periods=52).min()) / 2)
    # Shift both spans forward by 26 periods (we'll handle alignment differently)
    senkou_a = senkou_a.values
    senkou_b = senkou_b.values
    
    # Kumo (Cloud) boundaries: Senkou Span A and B
    # For cloud color: green when Senkou A > Senkou B (bullish), red when Senkou A < Senkou B (bearish)
    # We'll use the current cloud (values from 26 periods ago) for support/resistance
    
    for i in range(52, n):  # Start after 52-bar warmup for Senkou Span B
        # Skip if any required data is invalid
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(weekly_sma_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        
        # Ichimoku signals
        tenkan = tenkan_sen[i]
        kijun = kijun_sen[i]
        senkou_a_val = senkou_a[i]
        senkou_b_val = senkou_b[i]
        
        # Kumo (Cloud) top and bottom
        kumotop = max(senkou_a_val, senkou_b_val)
        kumobottom = min(senkou_a_val, senkou_b_val)
        
        # Kumo twist (color): green if Senkou A > Senkou B (bullish), red if Senkou A < Senkou B (bearish)
        kumo_bullish = senkou_a_val > senkou_b_val
        
        # Weekly trend filter
        weekly_trend_up = weekly_sma_50_aligned[i] > 0 and price_close > weekly_sma_50_aligned[i]
        weekly_trend_down = weekly_sma_50_aligned[i] > 0 and price_close < weekly_sma_50_aligned[i]
        
        # Tenkan/Kijun cross signals (using current vs previous)
        tenkan_prev = tenkan_sen[i-1]
        kijun_prev = kijun_sen[i-1]
        
        tk_cross_up = tenkan > kijun and tenkan_prev <= kijun_prev  # Crossed up
        tk_cross_down = tenkan < kijun and tenkan_prev >= kijun_prev  # Crossed down
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: TK cross up + price above Kumo + weekly uptrend
        if tk_cross_up and price_close > kumotop and weekly_trend_up:
            enter_long = True
        
        # Short: TK cross down + price below Kumo + weekly downtrend
        if tk_cross_down and price_close < kumobottom and weekly_trend_down:
            enter_short = True
        
        # Exit conditions: TK cross reverse OR price crosses Kumo in opposite direction
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if TK cross down OR price falls below Kumo bottom
            exit_long = tk_cross_down or (price_close < kumobottom)
        elif position == -1:
            # Exit short if TK cross up OR price rises above Kumo top
            exit_short = tk_cross_up or (price_close > kumotop)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
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