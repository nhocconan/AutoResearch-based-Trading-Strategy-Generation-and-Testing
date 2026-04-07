#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Breakout with Volume and ADX Filter
# Hypothesis: Price breaking above/below the 20-period Donchian channel indicates
# strong momentum. Volume confirms institutional participation, and ADX > 25
# filters for trending markets. Works in both bull and bear markets by taking
# breakouts in the direction of the trend.
# Target: 20-50 trades/year (80-200 over 4 years).

name = "4h_donchian_breakout_volume_adx_v1"
timeframe = "4h"
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
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate 200 EMA on daily timeframe for trend filter
    daily_close = df_daily['close'].values
    daily_close_series = pd.Series(daily_close)
    daily_ema_200 = daily_close_series.ewm(span=200, min_periods=200, adjust=False).mean().values
    daily_ema_200_aligned = align_htf_to_ltf(prices, df_daily, daily_ema_200)
    
    # Calculate 20-period Donchian channel on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate ADX for trend strength
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            elif minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(high)
        atr[period] = np.nansum(tr[1:period+1]) / period
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        
        for i in range(period, len(high)):
            if atr[i] != 0:
                plus_di[i] = 100 * (np.nansum(plus_dm[i-period+1:i+1]) / period) / atr[i]
                minus_di[i] = 100 * (np.nansum(minus_dm[i-period+1:i+1]) / period) / atr[i]
                if plus_di[i] + minus_di[i] != 0:
                    dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.zeros_like(high)
        adx[2*period-1] = np.nansum(dx[period:2*period]) / period
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(daily_ema_200_aligned[i]) or np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below Donchian low or trend weakens or volume drops
            if (low[i] < donchian_low[i] or adx[i] < 25 or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above Donchian high or trend weakens or volume drops
            if (high[i] > donchian_high[i] or adx[i] < 25 or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above Donchian high with volume, strong trend, and above daily EMA200
            if (high[i] > donchian_high[i] and adx[i] > 25 and vol_filter[i] and close[i] > daily_ema_200_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low with volume, strong trend, and below daily EMA200
            elif (low[i] < donchian_low[i] and adx[i] > 25 and vol_filter[i] and close[i] < daily_ema_200_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals