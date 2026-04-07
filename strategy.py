#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian Breakout + 1d Volume + ADX Trend Filter
# Hypothesis: Donchian(20) breakouts on 12h with volume confirmation and ADX trend strength filter.
# Works in bull/bear by only taking breakouts in direction of strong trend (ADX > 25).
# Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_donchian_breakout_1d_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Donchian Channel (20) on 12h
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # 1d Volume MA(20) for confirmation
    volume_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20)
    
    # 1d ADX(14) for trend strength
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] > minus_dm[i]:
                minus_dm[i] = 0
            elif minus_dm[i] > plus_dm[i]:
                plus_dm[i] = 0
            else:
                plus_dm[i] = 0
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        adx = np.zeros_like(high)
        
        atr[period] = np.nansum(tr[1:period+1]) / period
        plus_di[period] = np.nansum(plus_dm[1:period+1]) / atr[period] * 100 if atr[period] != 0 else 0
        minus_di[period] = np.nansum(minus_dm[1:period+1]) / atr[period] * 100 if atr[period] != 0 else 0
        
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_di[i] = (plus_di[i-1] * (period-1) + plus_dm[i]) / atr[i] * 100 if atr[i] != 0 else 0
            minus_di[i] = (minus_di[i-1] * (period-1) + minus_dm[i]) / atr[i] * 100 if atr[i] != 0 else 0
            dx[i] = abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) * 100 if (plus_di[i] + minus_di[i]) != 0 else 0
        
        adx[period] = np.nansum(dx[1:period+1]) / period
        for i in range(period+1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_14 = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma_20_aligned[i]) or np.isnan(adx_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 20-period average
        volume_conf = volume[i] > volume_ma_20_aligned[i]
        
        # Trend strength: ADX > 25 indicates strong trend
        trend_filter = adx_14_aligned[i] > 25
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower or trend weakens
            if close[i] < lowest_low[i] or adx_14_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper or trend weakens
            if close[i] > highest_high[i] or adx_14_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if volume_conf and trend_filter:
                # Breakout above upper band
                if close[i] > highest_high[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakdown below lower band
                elif close[i] < lowest_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals