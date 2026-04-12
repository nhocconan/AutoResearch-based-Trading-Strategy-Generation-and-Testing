#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot + volume spike + choppiness regime filter
    # Long when price touches Camarilla L3 in bullish chop regime (ADX<20)
    # Short when price touches Camarilla H3 in bearish chop regime (ADX<20)
    # Uses 1d HTF for Camarilla levels, 12h for regime filter
    # Discrete sizing 0.25 to minimize fee churn. Target: 12-30 trades/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels
    camarilla_h3 = np.zeros(len(close_1d))
    camarilla_l3 = np.zeros(len(close_1d))
    camarilla_h4 = np.zeros(len(close_1d))
    camarilla_l4 = np.zeros(len(close_1d))
    camarilla_h5 = np.zeros(len(close_1d))
    camarilla_l5 = np.zeros(len(close_1d))
    camarilla_h6 = np.zeros(len(close_1d))
    camarilla_l6 = np.zeros(len(close_1d))
    
    for i in range(len(close_1d)):
        if i < 1:
            camarilla_h3[i] = camarilla_l3[i] = np.nan
            camarilla_h4[i] = camarilla_l4[i] = np.nan
            camarilla_h5[i] = camarilla_l5[i] = np.nan
            camarilla_h6[i] = camarilla_l6[i] = np.nan
            continue
            
        high_val = high_1d[i-1]
        low_val = low_1d[i-1]
        close_val = close_1d[i-1]
        diff = high_val - low_val
        
        camarilla_h3[i] = close_val + 1.1 * diff / 6
        camarilla_l3[i] = close_val - 1.1 * diff / 6
        camarilla_h4[i] = close_val + 1.1 * diff / 4
        camarilla_l4[i] = close_val - 1.1 * diff / 4
        camarilla_h5[i] = close_val + 1.1 * diff / 2
        camarilla_l5[i] = close_val - 1.1 * diff / 2
        camarilla_h6[i] = close_val + 1.1 * diff
        camarilla_l6[i] = close_val - 1.1 * diff
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    camarilla_l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    camarilla_h6_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h6)
    camarilla_l6_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l6)
    
    # Get 12h data for regime filter (ADX)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ADX(14)
    def calculate_adx(high, low, close, period=14):
        n = len(high)
        tr = np.zeros(n)
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
        
        # Wilder's smoothing
        atr = np.zeros(n)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, n):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros(n)
        minus_di = np.zeros(n)
        dx = np.zeros(n)
        
        for i in range(period, n):
            if atr[i] > 0:
                plus_di[i] = 100 * (np.mean(plus_dm[i-period+1:i+1]) / atr[i])
                minus_di[i] = 100 * (np.mean(minus_dm[i-period+1:i+1]) / atr[i])
                if plus_di[i] + minus_di[i] > 0:
                    dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.zeros(n)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, n):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Calculate 12h EMA20 for trend filter
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # Calculate volume spike filter (12h)
    volume_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_12h * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(ema20_12h_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_ma_12h[i])):
            signals[i] = 0.0
            continue
        
        # Determine 12h regime
        chop_regime = adx_12h_aligned[i] < 20  # Choppy/ranging market
        trending = adx_12h_aligned[i] >= 20
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Entry logic - only in choppy regime with volume spike
        long_entry = False
        short_entry = False
        
        if chop_regime and vol_confirm:
            # Long when price touches or crosses above L3
            if close[i] >= camarilla_l3_aligned[i]:
                long_entry = True
            # Short when price touches or crosses below H3
            if close[i] <= camarilla_h3_aligned[i]:
                short_entry = True
        
        # Exit logic - reverse signal or regime change to trending
        long_exit = (not chop_regime) or (close[i] <= camarilla_l4_aligned[i])
        short_exit = (not chop_regime) or (close[i] >= camarilla_h4_aligned[i])
        
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

name = "12h_1d_camarilla_chop_volume_v1"
timeframe = "12h"
leverage = 1.0