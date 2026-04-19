#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with weekly ADX filter and daily volume confirmation.
# Long when price breaks above 20-period Donchian high and weekly ADX > 25 and daily volume > 1.5x 20-day average.
# Short when price breaks below 20-period Donchian low and weekly ADX > 25 and daily volume > 1.5x 20-day average.
# Exit when price crosses the 12-period EMA (trend filter) or reaches opposite Donchian band.
# Uses Donchian for trend structure, ADX for trend strength, volume for confirmation.
# Target: 15-30 trades/year per symbol to stay within frequency limits.
name = "12h_Donchian20_ADX_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for volume average
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_20d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20d)
    
    # Get weekly data for ADX
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1]) if high[i] - high[i-1] > high[i-1] - low[i] else 0
            minus_dm[i] = max(0, low[i-1] - low[i]) if high[i-1] - low[i] > high[i] - high[i-1] else 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth TR, +DM, -DM
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        
        # Initial values
        if len(high) >= period:
            atr[period-1] = np.mean(tr[1:period])
            plus_dm_sum = np.sum(plus_dm[1:period])
            minus_dm_sum = np.sum(minus_dm[1:period])
            plus_di[period-1] = 100 * plus_dm_sum / atr[period-1] if atr[period-1] != 0 else 0
            minus_di[period-1] = 100 * minus_dm_sum / atr[period-1] if atr[period-1] != 0 else 0
        
        # Wilder's smoothing
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_di[i] = 100 * ((plus_di[i-1] * (period-1) + plus_dm[i]) / atr[i]) if atr[i] != 0 else 0
            minus_di[i] = 100 * ((minus_di[i-1] * (period-1) + minus_dm[i]) / atr[i]) if atr[i] != 0 else 0
        
        # Calculate DX and ADX
        dx = np.zeros_like(high)
        adx = np.zeros_like(high)
        
        for i in range(period, len(high)):
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) if (plus_di[i] + minus_di[i]) != 0 else 0
        
        # Initial ADX value
        if len(high) >= 2*period:
            adx[2*period-1] = np.mean(dx[period:2*period])
        
        # Wilder's smoothing for ADX
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_14_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_14_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_14_1w)
    
    # Calculate Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12-period EMA for exit
    ema_12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 28)  # Ensure Donchian and ADX are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema_12[i]) or np.isnan(adx_14_1w_aligned[i]) or 
            np.isnan(vol_ma_20d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donch_high[i]
        lower = donch_low[i]
        ema = ema_12[i]
        adx = adx_14_1w_aligned[i]
        vol_ma = vol_ma_20d_aligned[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        # Trend strength filter
        trend_strong = adx > 25
        
        if position == 0:
            # Long entry: price breaks above Donchian high with trend strength and volume confirmation
            if price > upper and trend_strong and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low with trend strength and volume confirmation
            elif price < lower and trend_strong and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below EMA or reaches Donchian low
            if price < ema or price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above EMA or reaches Donchian high
            if price > ema or price > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals