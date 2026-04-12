#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Williams %R + 1w ADX regime filter
    # Bull regime: 1w ADX > 25 + price > 1w EMA50 → long when Williams %R < -80 (oversold)
    # Bear regime: 1w ADX > 25 + price < 1w EMA50 → short when Williams %R > -20 (overbought)
    # Range regime: 1w ADX < 20 → fade at Williams %R extremes (%R < -80 for long, %R > -20 for short)
    # Uses discrete sizing 0.25 to minimize fee churn. Target: 15-25 trades/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data for regime and trend filters
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1w ADX(14)
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
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate 1w EMA50
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 1d Williams %R(14)
    def calculate_williams_r(high, low, close, period=14):
        n = len(high)
        highest_high = np.zeros(n)
        lowest_low = np.zeros(n)
        williams_r = np.full(n, np.nan)
        
        for i in range(n):
            start_idx = max(0, i - period + 1)
            highest_high[i] = np.max(high[start_idx:i+1])
            lowest_low[i] = np.min(low[start_idx:i+1])
        
        for i in range(period-1, n):
            if highest_high[i] != lowest_low[i]:
                williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
        
        return williams_r
    
    williams_r_1d = calculate_williams_r(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(adx_1w_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(williams_r_1d[i])):
            signals[i] = 0.0
            continue
        
        # Determine 1w regime
        strong_trend = adx_1w_aligned[i] > 25
        ranging = adx_1w_aligned[i] < 20
        bullish_trend = strong_trend and (close[i] > ema50_1w_aligned[i])
        bearish_trend = strong_trend and (close[i] < ema50_1w_aligned[i])
        
        # Entry logic
        long_entry = False
        short_entry = False
        
        if bullish_trend:
            # Bull regime: long on oversold conditions
            long_entry = williams_r_1d[i] < -80
        elif bearish_trend:
            # Bear regime: short on overbought conditions
            short_entry = williams_r_1d[i] > -20
        elif ranging:
            # Range regime: mean reversion at Williams %R extremes
            long_entry = williams_r_1d[i] < -80  # Oversold bounce
            short_entry = williams_r_1d[i] > -20  # Overbought rejection
        
        # Exit logic: reverse signal or regime change
        long_exit = (bearish_trend and williams_r_1d[i] > -20) or ranging
        short_exit = (bullish_trend and williams_r_1d[i] < -80) or ranging
        
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

name = "1d_1w_williams_r_adx_regime_v1"
timeframe = "1d"
leverage = 1.0