#!/usr/bin/env python3
# 4h_camarilla_1d_pivot_v3
# Hypothesis: 4h strategy using 1d Camarilla pivot levels (H4/L4) for structure, 4h volume confirmation, and ADX regime filter.
# Long when price touches L4 with bullish volume/ADX>25; short when price touches H4 with bearish volume/ADX>25.
# Works in bull/bear: pivots act as support/resistance in ranging markets, ADX filters trend strength.
# Discrete sizing (±0.25) to minimize fee churn. Target: 75-200 total trades over 4 years (19-50/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_1d_pivot_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each 1d bar
    # Pivot = (H+L+C)/3
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    # H4 = Pivot + 1.1 * (High - Low) / 2
    # L4 = Pivot - 1.1 * (High - Low) / 2
    h4 = pivot + 1.1 * range_1d / 2.0
    l4 = pivot - 1.1 * range_1d / 2.0
    
    # Align 1d Camarilla levels to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # 4h ADX for regime filter (trending >25)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * (np.zeros_like(plus_dm))
        minus_di = 100 * (np.zeros_like(minus_dm))
        plus_smooth = np.zeros_like(plus_dm)
        minus_smooth = np.zeros_like(minus_dm)
        
        plus_smooth[period] = np.mean(plus_dm[1:period+1])
        minus_smooth[period] = np.mean(minus_dm[1:period+1])
        
        for i in range(period+1, len(plus_dm)):
            plus_smooth[i] = (plus_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_smooth[i] = (minus_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        plus_di = 100 * plus_smooth / (atr + 1e-10)
        minus_di = 100 * minus_smooth / (atr + 1e-10)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = np.zeros_like(dx)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # 4h volume confirmation: current volume > 1.3x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(adx[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes above H4 (take profit) or ADX < 20 (trend weak)
            if close[i] > h4_aligned[i] or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes below L4 (take profit) or ADX < 20 (trend weak)
            if close[i] < l4_aligned[i] or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation and ADX > 25 (trending regime)
            volume_confirmed = volume[i] > 1.3 * volume_ma[i]
            strong_trend = adx[i] > 25
            
            if volume_confirmed and strong_trend:
                # Long: price touches or crosses above L4 with bullish momentum
                if close[i] > l4_aligned[i] and close[i-1] <= l4_aligned[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short: price touches or crosses below H4 with bearish momentum
                elif close[i] < h4_aligned[i] and close[i-1] >= h4_aligned[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals