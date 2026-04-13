#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot levels from 1d + 1d volume spike (1.8x) + 1d ADX>20.
    # Long when price touches/pierces Camarilla H3 level with volume confirmation and trend.
    # Short when price touches/pierces Camarilla L3 level with volume confirmation and trend.
    # Exit when price moves back toward Camarilla PIVOT level (mean reversion to pivot).
    # Discrete size 0.25 to minimize fee churn. Target: 80-150 total trades over 4 years.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point (PP) = (High + Low + Close) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    
    # Range = High - Low
    rng = high_1d - low_1d
    
    # Camarilla levels
    h3 = pp + (rng * 1.1 / 4)  # Resistance level 3
    l3 = pp - (rng * 1.1 / 4)  # Support level 3
    h4 = pp + (rng * 1.1 / 2)  # Resistance level 4 (stop loss)
    l4 = pp - (rng * 1.1 / 2)  # Support level 4 (stop loss)
    
    # Calculate 1d volume mean (20-period) with min_periods
    volume_1d = df_1d['volume'].values
    volume_series = pd.Series(volume_1d)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ADX (14-period) with min_periods
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
        atr[period] = np.nansum(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        for i in range(period, len(high)):
            plus_di[i] = 100 * (plus_dm[i] / atr[i]) if atr[i] != 0 else 0
            minus_di[i] = 100 * (minus_dm[i] / atr[i]) if atr[i] != 0 else 0
        
        dx = np.zeros_like(high)
        for i in range(period, len(high)):
            if (plus_di[i] + minus_di[i]) != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.zeros_like(high)
        adx[2*period-1] = np.nansum(dx[period:2*period])
        for i in range(2*period, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align HTF indicators to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(pp_aligned[i]) or np.isnan(vol_ma_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume (aligned)
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        
        # Volume filter: current 1d volume > 1.8 * 20-period mean (volume spike)
        volume_confirmation = vol_1d_aligned[i] > 1.8 * vol_ma_aligned[i]
        
        # ADX filter: trending market (ADX > 20)
        trending_market = adx_aligned[i] > 20
        
        # Entry conditions: price touches/pierces Camarilla H3/L3 with filters
        long_entry = (close[i] >= h3_aligned[i] and 
                     volume_confirmation and 
                     trending_market)
        short_entry = (close[i] <= l3_aligned[i] and 
                      volume_confirmation and 
                      trending_market)
        
        # Exit conditions: price moves back toward pivot level (mean reversion)
        long_exit = close[i] <= pp_aligned[i]
        short_exit = close[i] >= pp_aligned[i]
        
        # Stoploss: price touches H4/L4 levels
        long_stop = close[i] >= h4_aligned[i]
        short_stop = close[i] <= l4_aligned[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and (long_exit or long_stop):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (short_exit or short_stop):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_pivot_volume_adx_v1"
timeframe = "4h"
leverage = 1.0