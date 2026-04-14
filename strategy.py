#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Vortex Indicator with weekly ADX trend filter and volume confirmation
# Vortex identifies trend direction via +VI/-VI crossovers. Weekly ADX > 25 ensures
# strong trend context, avoiding whipsaws in ranging markets. Volume spike confirms
# institutional participation. Works in bull/bear by only taking trades in direction
# of weekly trend (long when +VI > -VI and weekly trend up, short when -VI > +VI and weekly trend down).
# Target: 20-50 total trades over 4 years (5-12/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly ADX(14) for trend strength filter
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            high_diff = high[i] - high[i-1]
            low_diff = low[i-1] - low[i]
            
            plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
            minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(high)
        atr[period] = np.nansum(tr[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * np.zeros_like(high)
        minus_di = 100 * np.zeros_like(high)
        
        for i in range(period, len(high)):
            if atr[i] != 0:
                plus_di[i] = 100 * (plus_dm[i] / atr[i])
                minus_di[i] = 100 * (minus_dm[i] / atr[i])
        
        dx = np.zeros_like(high)
        for i in range(period, len(high)):
            if (plus_di[i] + minus_di[i]) != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.zeros_like(high)
        adx[2*period-1] = np.nanmean(dx[period:2*period])
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate Vortex Indicator on 4h
    def calculate_vortex(high, low, close, period=14):
        vm_plus = np.zeros_like(high)
        vm_minus = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            vm_plus[i] = abs(high[i] - low[i-1])
            vm_minus[i] = abs(low[i] - high[i-1])
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Sum over period
        vm_plus_sum = np.zeros_like(high)
        vm_minus_sum = np.zeros_like(high)
        tr_sum = np.zeros_like(high)
        
        for i in range(len(high)):
            if i < period:
                vm_plus_sum[i] = np.nansum(vm_plus[1:i+1]) if i > 0 else 0
                vm_minus_sum[i] = np.nansum(vm_minus[1:i+1]) if i > 0 else 0
                tr_sum[i] = np.nansum(tr[1:i+1]) if i > 0 else 0
            else:
                vm_plus_sum[i] = vm_plus_sum[i-1] - vm_plus_sum[i-period]/period + vm_plus[i]
                vm_minus_sum[i] = vm_minus_sum[i-1] - vm_minus_sum[i-period]/period + vm_minus[i]
                tr_sum[i] = tr_sum[i-1] - tr_sum[i-period]/period + tr[i]
        
        vi_plus = np.zeros_like(high)
        vi_minus = np.zeros_like(high)
        
        for i in range(len(high)):
            if tr_sum[i] != 0:
                vi_plus[i] = vm_plus_sum[i] / tr_sum[i]
                vi_minus[i] = vm_minus_sum[i] / tr_sum[i]
        
        return vi_plus, vi_minus
    
    vi_plus, vi_minus = calculate_vortex(high, low, close, 14)
    
    # Volume confirmation: volume > 2.0x average volume (28-period for 4h = 7 days)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=28, min_periods=28).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 28  # for volume average and vortex
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or 
            np.isnan(adx_1w_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Weekly trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_1w_aligned[i] > 25
        
        if position == 0:
            # Long: +VI crosses above -VI with volume filter AND strong weekly trend up (+VI > -VI)
            if (vi_plus[i] > vi_minus[i] and vi_plus[i-1] <= vi_minus[i-1] and 
                strong_trend and vi_plus[i] > vi_minus[i] and 
                vol > 2.0 * avg_vol[i]):
                position = 1
                signals[i] = position_size
            # Short: -VI crosses above +VI with volume filter AND strong weekly trend down (-VI > +VI)
            elif (vi_minus[i] > vi_plus[i] and vi_minus[i-1] <= vi_plus[i-1] and 
                  strong_trend and vi_minus[i] > vi_plus[i] and 
                  vol > 2.0 * avg_vol[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: -VI crosses above +VI OR weekly trend weakens
            if (vi_minus[i] > vi_plus[i] and vi_minus[i-1] <= vi_plus[i-1]) or not strong_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: +VI crosses above -VI OR weekly trend weakens
            if (vi_plus[i] > vi_minus[i] and vi_plus[i-1] <= vi_minus[i-1]) or not strong_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Vortex_ADX_Volume_Trend"
timeframe = "4h"
leverage = 1.0