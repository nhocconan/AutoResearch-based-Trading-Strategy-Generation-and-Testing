#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Camarilla pivot breakout from 1w with volume confirmation and ADX trend filter.
    # Long when price breaks above Camarilla H3 level with volume spike and ADX>25.
    # Short when price breaks below Camarilla L3 level with volume spike and ADX>25.
    # Exit when price returns to Camarilla pivot point (mean reversion).
    # Discrete size 0.25 to minimize fee churn. Target: 30-100 total trades over 4 years.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w OHLC for Camarilla pivots
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    # Camarilla pivot levels (based on previous week)
    # Pivot = (High + Low + Close) / 3
    # Range = High - Low
    # H3 = Close + Range * 1.1 / 4
    # L3 = Close - Range * 1.1 / 4
    # H4 = Close + Range * 1.1 / 2
    # L4 = Close - Range * 1.1 / 2
    
    pivot = (high_1w + low_1w + close_1w) / 3.0
    rng = high_1w - low_1w
    
    # H3 and L3 are the key breakout levels
    camarilla_h3 = close_1w + rng * 1.1 / 4.0
    camarilla_l3 = close_1w - rng * 1.1 / 4.0
    camarilla_pivot = pivot  # Exit level
    
    # Calculate 1w volume mean (20-period) with min_periods
    volume_1w = df_1w['volume'].values
    volume_series = pd.Series(volume_1w)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w ADX (14-period) with min_periods
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
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    
    # Align HTF indicators to 1d timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1w, camarilla_pivot)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(vol_ma_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1w volume (aligned)
        vol_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w)
        
        # Volume filter: current 1w volume > 2.0 * 20-period mean (volume spike)
        volume_confirmation = vol_1w_aligned[i] > 2.0 * vol_ma_aligned[i]
        
        # ADX filter: strong trending market (ADX > 25)
        strong_trend = adx_aligned[i] > 25
        
        # Entry conditions: price breaks Camarilla H3/L3 levels with filters
        long_entry = (close[i] > camarilla_h3_aligned[i] and 
                     volume_confirmation and 
                     strong_trend)
        short_entry = (close[i] < camarilla_l3_aligned[i] and 
                      volume_confirmation and 
                      strong_trend)
        
        # Exit conditions: price returns to Camarilla pivot point (mean reversion)
        long_exit = close[i] < camarilla_pivot_aligned[i]
        short_exit = close[i] > camarilla_pivot_aligned[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
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

name = "1d_1w_camarilla_breakout_volume_adx_v1"
timeframe = "1d"
leverage = 1.0