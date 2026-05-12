#!/usr/bin/env python3
# 1D_WEEKLY_ICHIMOKU_KUMO_BREAKOUT_1W_TREND_FILTER
# Hypothesis: Ichimoku Cloud (Tenkan-sen/Kijun-sen/Senkou Span A/B) on weekly timeframe
# identifies strong support/resistance zones. Price breaking above/below the cloud
# with Tenkan/Kijun crossover signals strong momentum. Combined with weekly ADX
# trend filter (>25) to avoid false breakouts in chop. Works in bull markets
# (price > cloud + bullish TK cross) and bear markets (price < cloud + bearish TK cross).
# Target: 15-25 trades/year on 1d timeframe (60-100 total over 4 years).

name = "1D_WEEKLY_ICHIMOKU_KUMO_BREAKOUT_1W_TREND_FILTER"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Weekly data for Ichimoku calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:  # Need at least 1 year of weekly data
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    high_9 = pd.Series(high_1w).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_9 = pd.Series(low_1w).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    high_26 = pd.Series(high_1w).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_26 = pd.Series(low_1w).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    high_52 = pd.Series(high_1w).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_52 = pd.Series(low_1w).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = (high_52 + low_52) / 2
    
    # Chikou Span (Lagging Span): current close plotted 26 periods back
    # Not used for signals but calculated for completeness
    
    # Weekly ADX for trend filter
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0) if high[i] - high[i-1] > low[i-1] - low[i] else 0
            minus_dm[i] = max(low[i-1] - low[i], 0) if low[i-1] - low[i] > high[i] - high[i-1] else 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth with Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(high)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * (np.zeros_like(high))
        minus_di = 100 * (np.zeros_like(high))
        dx = 100 * (np.zeros_like(high))
        
        # Smooth +DM and -DM
        smoothed_plus_dm = np.zeros_like(high)
        smoothed_minus_dm = np.zeros_like(high)
        smoothed_plus_dm[period] = np.sum(plus_dm[1:period+1])
        smoothed_minus_dm[period] = np.sum(minus_dm[1:period+1])
        
        for i in range(period+1, len(high)):
            smoothed_plus_dm[i] = smoothed_plus_dm[i-1] - (smoothed_plus_dm[i-1] / period) + plus_dm[i]
            smoothed_minus_dm[i] = smoothed_minus_dm[i-1] - (smoothed_minus_dm[i-1] / period) + minus_dm[i]
        
        # Calculate DI and DX
        for i in range(period, len(high)):
            if atr[i] != 0:
                plus_di[i] = 100 * smoothed_plus_dm[i] / atr[i]
                minus_di[i] = 100 * smoothed_minus_dm[i] / atr[i]
                if plus_di[i] + minus_di[i] != 0:
                    dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        # ADX: smoothed DX
        adx = np.zeros_like(high)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    
    # Align to daily timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Need enough weekly data for Senkou Span B
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(adx_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A/B)
        upper_cloud = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        if position == 0:
            # LONG: Price above cloud + bullish TK cross + ADX > 25
            if (close[i] > upper_cloud and 
                tenkan_sen_aligned[i] > kijun_sen_aligned[i] and 
                adx_1w_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below cloud + bearish TK cross + ADX > 25
            elif (close[i] < lower_cloud and 
                  tenkan_sen_aligned[i] < kijun_sen_aligned[i] and 
                  adx_1w_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below cloud OR bearish TK cross OR ADX < 20
            if (close[i] < lower_cloud or 
                tenkan_sen_aligned[i] < kijun_sen_aligned[i] or 
                adx_1w_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above cloud OR bullish TK cross OR ADX < 20
            if (close[i] > upper_cloud or 
                tenkan_sen_aligned[i] > kijun_sen_aligned[i] or 
                adx_1w_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals