#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d weekly pivot direction filter
# Long when: price > Kumo cloud AND Tenkan > Kijun AND weekly pivot bias bullish
# Short when: price < Kumo cloud AND Tenkan < Kijun AND weekly pivot bias bearish
# Uses Kumo twist for trend confirmation and weekly pivot for higher timeframe direction
# Discrete position size 0.25. Target: 50-150 total trades over 4 years (12-37/year).
# Works in both bull and bear by requiring alignment of multiple timeframes and trend filters.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 6h Ichimoku Components (9, 26, 52 periods) ===
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2 plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 plotted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close plotted 26 periods behind
    # Not used for entry as it requires future data
    
    # Kumo (Cloud) boundaries: Senkou Span A and B
    # For cloud twist, we need current cloud and prior cloud
    # Since Senkou spans are plotted 26 periods ahead, we look back 26 periods for current values
    senkou_a_current = np.roll(senkou_a, 26)
    senkou_b_current = np.roll(senkou_b, 26)
    senkou_a_prior = np.roll(senkou_a_current, 1)  # Prior period cloud
    senkou_b_prior = np.roll(senkou_b_current, 1)
    
    # Handle NaN from rolling and rolling
    warmup_ichimoku = 52 + 26  # 52 for Senkou B calculation + 26 for forward shift
    
    # === 1d Weekly Pivot Points (from weekly data) ===
    df_1w = get_htf_data(prices, '1w')
    # Typical Price = (High + Low + Close) / 3
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    # Weekly Pivot Point (PP)
    pp = typical_price.values
    # Resistance 1 (R1) = (2 * PP) - Low
    r1 = (2 * pp) - df_1w['low'].values
    # Support 1 (S1) = (2 * PP) - High
    s1 = (2 * pp) - df_1w['high'].values
    # Resistance 2 (R2) = PP + (High - Low)
    r2 = pp + (df_1w['high'].values - df_1w['low'].values)
    # Support 2 (S2) = PP - (High - Low)
    s2 = pp - (df_1w['high'].values - df_1w['low'].values)
    
    # Align weekly pivot data to 6h timeframe (completed weekly bar only)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Weekly pivot bias: bullish if price > PP, bearish if price < PP
    # We'll use the 6h close price vs weekly PP
    # Need to align 6h close to weekly timeframe for comparison, but simpler:
    # Use weekly PP as reference - if 6h price is above weekly PP, bullish bias
    # Since we can't get 6h close in weekly alignment without lookahead,
    # we use the weekly PP level as a static reference for the week
    weekly_bias_bullish = close > pp_aligned  # Simplified: price above weekly PP = bullish
    weekly_bias_bearish = close < pp_aligned   # Price below weekly PP = bearish
    
    # Session filter: 08-20 UTC (avoid low volume Asian session)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(warmup_ichimoku, 50)  # Ichimoku needs 52+26=78, plus buffer
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a_current[i]) or 
            np.isnan(senkou_b_current[i]) or np.isnan(pp_aligned[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        senkou_a_val = senkou_a_current[i]
        senkou_b_val = senkou_b_current[i]
        
        # Kumo cloud boundaries (top and bottom of cloud)
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        # Kumo twist: bullish when Senkou A > Senkou B, bearish when Senkou A < Senkou B
        kumo_twist_bullish = senkou_a_val > senkou_b_val
        kumo_twist_bearish = senkou_a_val < senkou_b_val
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price falls below cloud OR Tenkan crosses below Kijun
            if price < cloud_bottom or tenkan_val < kijun_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price rises above cloud OR Tenkan crosses above Kijun
            if price > cloud_top or tenkan_val > kijun_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price above cloud, Tenkan > Kijun, bullish Kumo twist, weekly bias bullish
            if (price > cloud_top and tenkan_val > kijun_val and 
                kumo_twist_bullish and weekly_bias_bullish[i]):
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price below cloud, Tenkan < Kijun, bearish Kumo twist, weekly bias bearish
            elif (price < cloud_bottom and tenkan_val < kijun_val and 
                  kumo_twist_bearish and weekly_bias_bearish[i]):
                signals[i] = -0.25
                position = -1
        
        else:
            # Maintain current position
            signals[i] = position * 0.25
    
    return signals

name = "6h_Ichimoku_WeeklyPivotBias_V1"
timeframe = "6h"
leverage = 1.0