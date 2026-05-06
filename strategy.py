#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Ichimoku cloud (Senkou Span A/B) + TK Cross for trend direction
# Entry when price breaks above/below cloud with TK cross confirmation and volume spike (>1.8x 20-bar avg)
# Exit when price re-enters cloud or TK cross reverses
# Uses discrete sizing 0.25 to balance return and drawdown; target 80-120 total trades over 4 years (20-30/year)
# Ichimoku works in both bull/bear markets by adapting to trend via cloud color and TK cross
# Weekly timeframe used only for regime filter: only trade when weekly close > weekly EMA20 (bull) or < weekly EMA20 (bear)
# This avoids counter-trend trades in strong weekly regimes while allowing mean-reversion in ranging weeks

name = "6h_Ichimoku_Cloud_TKCross_WeeklyRegime_v1"
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
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d Ichimoku components (standard periods: 9, 26, 52)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max()
    low_9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min()
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(high_1d).rolling(window=26, min_periods=26).max()
    low_26 = pd.Series(low_1d).rolling(window=26, min_periods=26).min()
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    high_52 = pd.Series(high_1d).rolling(window=52, min_periods=52).max()
    low_52 = pd.Series(low_1d).rolling(window=52, min_periods=52).min()
    senkou_b = ((high_52 + low_52) / 2).shift(26)
    
    # Chikou Span (Lagging Span): close shifted -26 periods (not used for signals)
    
    # Calculate weekly EMA20 for regime filter
    close_1w_series = pd.Series(close_1w)
    ema20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align HTF indicators to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan.values)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun.values)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a.values)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b.values)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w, additional_delay_bars=0)
    
    # Pre-compute volume spike filter (>1.8x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma_20)
    volume_filter_aligned = align_htf_to_ltf(prices, df_1d, volume_filter, additional_delay_bars=0)
    
    # Pre-compute session filter (08-20 UTC) - optional but helps reduce noise
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(ema20_1w_aligned[i]) or np.isnan(volume_filter_aligned[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud color and boundaries
        upper_cloud = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        lower_cloud = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # TK Cross: Tenkan > Kijun = bullish, Tenkan < Kijun = bearish
        tk_bullish = tenkan_aligned[i] > kijun_aligned[i]
        tk_bearish = tenkan_aligned[i] < kijun_aligned[i]
        
        # Weekly regime: only trade in direction of weekly trend
        weekly_bull = close_1w.iloc[-1] > ema20_1w_aligned[i] if len(close_1w) > 0 else False
        weekly_bear = close_1w.iloc[-1] < ema20_1w_aligned[i] if len(close_1w) > 0 else False
        
        if position == 0:
            # Long: price above cloud, TK bullish, volume spike, weekly bullish regime
            if (close[i] > upper_cloud and tk_bullish and volume_filter_aligned[i] and weekly_bull):
                signals[i] = 0.25
                position = 1
            # Short: price below cloud, TK bearish, volume spike, weekly bearish regime
            elif (close[i] < lower_cloud and tk_bearish and volume_filter_aligned[i] and weekly_bear):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters cloud OR TK cross turns bearish
            if close[i] <= upper_cloud or not tk_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters cloud OR TK cross turns bullish
            if close[i] >= lower_cloud or tk_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals