#!/usr/bin/env python3
"""
4h_donchian_20_1d_trend_volume_v2
Hypothesis: On 4-hour timeframe, use Donchian(20) breakouts with trend filter from 1-day EMA200 and volume confirmation. Enter long on upper band breakout in uptrend with volume > 1.5x average, short on lower band breakdown in downtrend with volume > 1.5x average. Exit on opposite band touch. Add 1-day ADX filter (>25) to avoid chop. Designed for low frequency (<30 trades/year) to avoid fee drift while capturing trend continuation. Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend) by using 1-day trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_20_1d_trend_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend and ADX filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    d_close = df_1d['close'].values
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    
    # Calculate daily EMA200 for trend
    d_ema200 = pd.Series(d_close).ewm(span=200, adjust=False).mean().values
    d_ema200_aligned = align_htf_to_ltf(prices, df_1d, d_ema200)
    
    # Calculate daily ADX(14) for trend strength
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] > minus_dm[i]:
                minus_dm[i] = 0
            elif minus_dm[i] > plus_dm[i]:
                plus_dm[i] = 0
            else:
                plus_dm[i] = 0
                minus_dm[i] = 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth with Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        
        atr[period-1] = np.mean(tr[1:period+1]) if period < len(tr) else np.nan
        plus_dm_sum = np.sum(plus_dm[1:period+1]) if period < len(plus_dm) else np.nan
        minus_dm_sum = np.sum(minus_dm[1:period+1]) if period < len(minus_dm) else np.nan
        
        if period < len(tr) and not np.isnan(atr[period-1]):
            atr[period-1] = np.mean(tr[1:period+1])
            plus_di[period-1] = 100 * plus_dm_sum / (atr[period-1] * period) if atr[period-1] != 0 else 0
            minus_di[period-1] = 100 * minus_dm_sum / (atr[period-1] * period) if atr[period-1] != 0 else 0
            
            for i in range(period, len(high)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                plus_di[i] = 100 * (plus_di[i-1] * (period-1) + plus_dm[i]) / (atr[i] * period) if atr[i] != 0 else 0
                minus_di[i] = 100 * (minus_di[i-1] * (period-1) + minus_dm[i]) / (atr[i] * period) if atr[i] != 0 else 0
        
        dx = np.zeros_like(high)
        adx = np.zeros_like(high)
        for i in range(period, len(high)):
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
            else:
                dx[i] = 0
        
        adx[2*period-1] = np.mean(dx[period:2*period]) if 2*period < len(dx) else np.nan
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    d_adx = calculate_adx(d_high, d_low, d_close, 14)
    d_adx_aligned = align_htf_to_ltf(prices, df_1d, d_adx)
    
    # Calculate 40-period average volume for confirmation
    vol_avg = pd.Series(volume).rolling(window=40, min_periods=40).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if daily indicators not available
        if np.isnan(d_ema200_aligned[i]) or np.isnan(d_adx_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend based on price vs daily EMA200
        uptrend = close[i] > d_ema200_aligned[i]
        downtrend = close[i] < d_ema200_aligned[i]
        
        # Strong trend filter: ADX > 25
        strong_trend = d_adx_aligned[i] > 25
        
        # Volume confirmation: current volume > 1.5x 40-period average
        vol_confirm = volume[i] > 1.5 * vol_avg[i] if not np.isnan(vol_avg[i]) else False
        
        if position == 1:  # Long position
            # Exit when price touches or goes below lower Donchian(20)
            if i >= 20:
                donchian_low = np.min(low[i-20:i])
                if close[i] <= donchian_low:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price touches or goes above upper Donchian(20)
            if i >= 20:
                donchian_high = np.max(high[i-20:i])
                if close[i] >= donchian_high:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need at least 20 periods for Donchian calculation
            if i >= 20:
                donchian_high = np.max(high[i-20:i])
                donchian_low = np.min(low[i-20:i])
                
                # Long entry: price breaks above upper Donchian(20) in uptrend with volume confirmation and strong trend
                long_entry = (close[i] > donchian_high) and uptrend and vol_confirm and strong_trend
                # Short entry: price breaks below lower Donchian(20) in downtrend with volume confirmation and strong trend
                short_entry = (close[i] < donchian_low) and downtrend and vol_confirm and strong_trend
                
                if long_entry:
                    position = 1
                    signals[i] = 0.25
                elif short_entry:
                    position = -1
                    signals[i] = -0.25
    
    return signals