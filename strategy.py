#!/usr/bin/env python3
"""
6h_VolumeBreakout_1dTrend_1wRegime
Hypothesis: In 6-hour timeframe, volume-confirmed breakouts from prior period's high/low 
with daily trend filter (EMA34) and weekly regime filter (ADX) capture momentum. 
Weekly ADX > 25 filters for trending markets (both bull/bear), avoiding range-bound 
whipsaws. Uses discrete position sizing (0.25) to limit trade frequency and 
minimize fee drift. Target: 20-40 trades/year across BTC/ETH/SOL.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close']
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    
    # Get weekly data for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high']
    low_1w = df_1w['low']
    close_1w = df_1w['close']
    
    # Daily EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Weekly ADX(14) for regime filter
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            tr[i] = max(high[i] - low[i], 
                       abs(high[i] - close[i-1]), 
                       abs(low[i] - close[i-1]))
        
        # Smooth using Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        
        # Smooth DM values
        plus_dm_smooth = np.zeros_like(plus_dm)
        minus_dm_smooth = np.zeros_like(minus_dm)
        plus_dm_smooth[period] = np.mean(plus_dm[1:period+1])
        minus_dm_smooth[period] = np.mean(minus_dm[1:period+1])
        
        for i in range(period+1, len(high)):
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
            
            if atr[i] != 0:
                plus_di[i] = (plus_dm_smooth[i] / atr[i]) * 100
                minus_di[i] = (minus_dm_smooth[i] / atr[i]) * 100
                if (plus_di[i] + minus_di[i]) != 0:
                    dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
        
        # Smooth DX to get ADX
        adx = np.zeros_like(high)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(adx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
            
        return adx
    
    adx_14_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_14_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_14_1w)
    
    # 6-period high/low for breakout levels (prior period)
    high_6 = pd.Series(high).rolling(window=6, min_periods=6).max().shift(1).values
    low_6 = pd.Series(low).rolling(window=6, min_periods=6).min().shift(1).values
    
    # Volume spike: 2.5x 24-period average (6h * 4 = 1 day)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(adx_14_1w_aligned[i]) or
            np.isnan(high_6[i]) or
            np.isnan(low_6[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_trend = ema_34_1d_aligned[i]
        adx_regime = adx_14_1w_aligned[i]
        breakout_high = high_6[i]
        breakout_low = low_6[i]
        vol_spike = volume_spike[i]
        
        # Only trade in trending regime (ADX > 25)
        if adx_regime > 25:
            if position == 0:
                # Long: break above 6-period high with volume spike and above daily EMA
                if price > breakout_high and vol_spike and price > ema_trend:
                    signals[i] = 0.25
                    position = 1
                # Short: break below 6-period low with volume spike and below daily EMA
                elif price < breakout_low and vol_spike and price < ema_trend:
                    signals[i] = -0.25
                    position = -1
            
            elif position == 1:
                # Maintain long position
                signals[i] = 0.25
                # Exit: price breaks below 6-period low or below daily EMA
                if price < breakout_low or price < ema_trend:
                    signals[i] = 0.0
                    position = 0
            
            elif position == -1:
                # Maintain short position
                signals[i] = -0.25
                # Exit: price breaks above 6-period high or above daily EMA
                if price > breakout_high or price > ema_trend:
                    signals[i] = 0.0
                    position = 0
    
    return signals

name = "6h_VolumeBreakout_1dTrend_1wRegime"
timeframe = "6h"
leverage = 1.0