#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with volume confirmation and ADX trend filter.
# Long when price breaks above 20-period Donchian high, ADX > 25, volume > 1.5x 20-period average.
# Short when price breaks below 20-period Donchian low, ADX > 25, volume > 1.5x 20-period average.
# Exit when price crosses the opposite Donchian band or ADX falls below 20.
# Uses discrete position size (0.25) to minimize churn. Designed for 4h timeframe
# to capture medium-term trends while avoiding whipsaws in both bull and bear markets.
# Target: 20-50 trades/year per symbol (~80-200 total over 4 years).
name = "4h_Donchian20_Volume_ADX_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
                
            tr[i] = max(high[i] - low[i], 
                       abs(high[i] - close[i-1]), 
                       abs(low[i] - close[i-1]))
        
        # Smooth using Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(tr)
        plus_dm_smooth = np.zeros_like(plus_dm)
        minus_dm_smooth = np.zeros_like(minus_dm)
        
        atr[period] = np.mean(tr[1:period+1])
        plus_dm_smooth[period] = np.mean(plus_dm[1:period+1])
        minus_dm_smooth[period] = np.mean(minus_dm[1:period+1])
        
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        dx = np.zeros_like(close)
        valid = (plus_di[period:] + minus_di[period:]) > 0
        dx[period:] = np.where(valid, 100 * np.abs(plus_di[period:] - minus_di[period:]) / (plus_di[period:] + minus_di[period:]), 0)
        
        adx = np.zeros_like(close)
        if len(dx) >= 2*period+1:
            adx[2*period] = np.mean(dx[period:2*period+1])
            for i in range(2*period+1, len(dx)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 28)  # Ensure Donchian and ADX are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_band = donchian_high[i]
        lower_band = donchian_low[i]
        adx_val = adx[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        # ADX trend strength filter
        strong_trend = adx_val > 25
        
        if position == 0:
            # Enter long if price breaks above upper band, strong trend, and volume confirmation
            if price > upper_band and strong_trend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short if price breaks below lower band, strong trend, and volume confirmation
            elif price < lower_band and strong_trend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price crosses below lower band or trend weakens
            if price < lower_band or adx_val < 20:  # Trend weakening
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price crosses above upper band or trend weakens
            if price > upper_band or adx_val < 20:  # Trend weakening
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals