#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + 12h ADX regime filter + volume confirmation
    # In bull regime (12h ADX>25 + price>EMA50): long on upper Donchian breakout
    # In bear regime (12h ADX>25 + price<EMA50): short on lower Donchian breakout
    # In range regime (12h ADX<20): fade at Donchian bands with volume confirmation
    # Uses discrete sizing 0.25 to minimize fee churn. Target: 20-40 trades/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
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
    
    # Calculate 12h EMA50
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate 4h Donchian channels (20-period)
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        upper[i] = np.max(high[i-lookback+1:i+1])
        lower[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine 12h regime
        strong_trend = adx_12h_aligned[i] > 25
        ranging = adx_12h_aligned[i] < 20
        bullish_trend = strong_trend and (close[i] > ema50_12h_aligned[i])
        bearish_trend = strong_trend and (close[i] < ema50_12h_aligned[i])
        
        # Entry logic
        long_entry = False
        short_entry = False
        
        if bullish_trend:
            # Bull regime: long on upper Donchian breakout with volume
            long_entry = (close[i] > upper[i-1]) and volume_spike[i]
        elif bearish_trend:
            # Bear regime: short on lower Donchian breakout with volume
            short_entry = (close[i] < lower[i-1]) and volume_spike[i]
        elif ranging:
            # Range regime: fade at Donchian bands with volume confirmation
            long_entry = (close[i] < lower[i]) and volume_spike[i]  # Oversold bounce
            short_entry = (close[i] > upper[i]) and volume_spike[i]  # Overbought rejection
        
        # Exit logic: reverse signal or volatility expansion
        long_exit = (bearish_trend and close[i] < lower[i]) or (adx_12h_aligned[i] < 15)
        short_exit = (bullish_trend and close[i] > upper[i]) or (adx_12h_aligned[i] < 15)
        
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

name = "4h_12h_donchian_adx_volume_v1"
timeframe = "4h"
leverage = 1.0