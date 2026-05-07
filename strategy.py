#!/usr/bin/env python3
# 6H_IchimokuCloud_WeeklyTrend_Filter
# Hypothesis: 6-hour Ichimoku cloud breakout with weekly trend filter (price above/below weekly Kumo) and volume confirmation.
# Uses weekly trend to avoid counter-trend trades in both bull and bear markets.
# Cloud breakout provides clear entry/exit signals. Volume spike ensures momentum confirmation.
# Targets 15-30 trades/year to minimize fee drift. Uses discrete position sizing (0.25).

name = "6H_IchimokuCloud_WeeklyTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:  # Need enough data for weekly calculations
        return np.zeros(n)
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    # Not used for entry as it requires future data
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), tenkan)
    kijun_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), senkou_b)
    
    # Calculate weekly trend filter: price relative to weekly Kumo (cloud)
    # Weekly Tenkan-sen and Kijun-sen
    wk_high = df_1w['high'].values
    wk_low = df_1w['low'].values
    wk_close = df_1w['close'].values
    
    wk_period9_high = pd.Series(wk_high).rolling(window=9, min_periods=9).max().values
    wk_period9_low = pd.Series(wk_low).rolling(window=9, min_periods=9).min().values
    wk_tenkan = (wk_period9_high + wk_period9_low) / 2
    
    wk_period26_high = pd.Series(wk_high).rolling(window=26, min_periods=26).max().values
    wk_period26_low = pd.Series(wk_low).rolling(window=26, min_periods=26).min().values
    wk_kijun = (wk_period26_high + wk_period26_low) / 2
    
    # Weekly Senkou Span A and B
    wk_senkou_a = (wk_tenkan + wk_kijun) / 2
    wk_period52_high = pd.Series(wk_high).rolling(window=52, min_periods=52).max().values
    wk_period52_low = pd.Series(wk_low).rolling(window=52, min_periods=52).min().values
    wk_senkou_b = (wk_period52_high + wk_period52_low) / 2
    
    # Align weekly components to 6h timeframe
    wk_tenkan_aligned = align_htf_to_ltf(prices, df_1w, wk_tenkan)
    wk_kijun_aligned = align_htf_to_ltf(prices, df_1w, wk_kijun)
    wk_senkou_a_aligned = align_htf_to_ltf(prices, df_1w, wk_senkou_a)
    wk_senkou_b_aligned = align_htf_to_ltf(prices, df_1w, wk_senkou_b)
    
    # Weekly Kumo top and bottom
    wk_kumo_top = np.maximum(wk_senkou_a_aligned, wk_senkou_b_aligned)
    wk_kumo_bottom = np.minimum(wk_senkou_a_aligned, wk_senkou_b_aligned)
    
    # Volume filter: current volume > 1.8x average volume (24-period for 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Volatility filter: avoid low volatility periods (ATR < 0.4% of price)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    vol_filter = atr > 0.004 * close  # ATR > 0.4% of price
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 24)  # Ensure we have enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or np.isnan(senkou_a_aligned[i]) or
            np.isnan(senkou_b_aligned[i]) or np.isnan(wk_tenkan_aligned[i]) or np.isnan(wk_kijun_aligned[i]) or
            np.isnan(wk_senkou_a_aligned[i]) or np.isnan(wk_senkou_b_aligned[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(vol_filter[i]) or not vol_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation (1.8x average volume)
        volume_filter = volume[i] > 1.8 * vol_ma[i]
        
        # Cloud conditions
        cloud_top = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Long: Price breaks above cloud + TK cross bullish + weekly uptrend (price above weekly Kumo) + volume spike
            if (close[i] > cloud_top and 
                tenkan_aligned[i] > kijun_aligned[i] and   # TK cross bullish
                close[i] > wk_kumo_top[i] and              # Price above weekly Kumo (uptrend)
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below cloud + TK cross bearish + weekly downtrend (price below weekly Kumo) + volume spike
            elif (close[i] < cloud_bottom and 
                  tenkan_aligned[i] < kijun_aligned[i] and   # TK cross bearish
                  close[i] < wk_kumo_bottom[i] and           # Price below weekly Kumo (downtrend)
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: Price returns to the cloud or TK cross reverses
            tk_cross_bullish = tenkan_aligned[i] > kijun_aligned[i]
            tk_cross_bearish = tenkan_aligned[i] < kijun_aligned[i]
            
            if position == 1:
                # Exit long: price falls below cloud or TK cross turns bearish
                if (close[i] < cloud_bottom or tk_cross_bearish):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price rises above cloud or TK cross turns bullish
                if (close[i] > cloud_top or tk_cross_bullish):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals