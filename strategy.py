#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) with 1d ADX regime filter and volume spike confirmation.
Long when Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND 1d ADX > 25 (trending) AND volume > 2.0x 20-period MA.
Short when Bull Power < 0 AND Bear Power > 0 (bearish momentum) AND 1d ADX > 25 (trending) AND volume > 2.0x 20-period MA.
Exit when momentum weakens (Bull Power and Bear Power same sign) or volume filter fails.
Elder Ray measures bull/bear power via EMA13; ADX filters for trending markets to avoid whipsaws.
Designed for ~15-25 trades/year with strong trend-following edge in both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d ADX for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ADX calculation (Wilder's smoothing)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0) if high[i] - high[i-1] > low[i-1] - low[i] else 0
            minus_dm[i] = max(low[i-1] - low[i], 0) if low[i-1] - low[i] > high[i] - high[i-1] else 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(tr)
        plus_dm_smooth = np.zeros_like(plus_dm)
        minus_dm_smooth = np.zeros_like(minus_dm)
        
        atr[period-1] = np.mean(tr[:period])
        plus_dm_smooth[period-1] = np.mean(plus_dm[:period])
        minus_dm_smooth[period-1] = np.mean(minus_dm[:period])
        
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
        minus_di = 100 * minus_dm_smooth / (atr + 1e-10)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        
        adx = np.zeros_like(dx)
        adx[2*period-1] = np.mean(dx[period-1:2*period-1])
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Elder Ray Index (Bull Power/Bear Power) on 6h
    ema_period = 13
    ema_close = pd.Series(close).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    bull_power = high - ema_close  # Bull Power = High - EMA
    bear_power = low - ema_close   # Bear Power = Low - EMA
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(ema_period, 20, 28)  # EMA13, volume MA20, ADX needs 2*period
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_close[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: 1d ADX > 25 = trending market
        trending = adx_1d_aligned[i] > 25
        
        # Volume filter: 6h volume > 2.0x 20-period MA
        vol_filter = volume[i] > 2.0 * vol_ma_20[i]
        
        # Elder Ray conditions
        bullish_momentum = bull_power[i] > 0 and bear_power[i] < 0  # Bull Power > 0 AND Bear Power < 0
        bearish_momentum = bull_power[i] < 0 and bear_power[i] > 0  # Bull Power < 0 AND Bear Power > 0
        momentum_weakening = (bull_power[i] > 0 and bear_power[i] > 0) or \
                             (bull_power[i] < 0 and bear_power[i] < 0)  # Same sign = weakening
        
        if position == 0:
            # Long: Bullish momentum AND trending AND volume confirmation
            if bullish_momentum and trending and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Bearish momentum AND trending AND volume confirmation
            elif bearish_momentum and trending and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: momentum weakening or volume filter fails
            exit_signal = momentum_weakening or not vol_filter
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_ADXRegime_VolumeSpike"
timeframe = "6h"
leverage = 1.0