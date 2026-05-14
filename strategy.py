#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d ADX trend filter and 4h volume confirmation.
# Long when price breaks above Donchian(20) upper band with 1d ADX > 25 and 4h volume > 1.5x 20-period average.
# Short when price breaks below Donchian(20) lower band with 1d ADX > 25 and 4h volume > 1.5x 20-period average.
# Exit on opposite Donchian band (lower for longs, upper for shorts).
# Uses discrete position sizing (0.30) to balance profit potential and drawdown control.
# Volume confirmation reduces false breakouts. ADX filter ensures trending markets only.
# Target: 80-150 total trades over 4 years = 20-38/year for 4h timeframe.
# Works in bull/bear: 1d ADX > 25 ensures strong trend, Donchian provides objective breakout levels.

name = "4h_Donchian20_Breakout_1dADX_4hVolumeConfirm"
timeframe = "4h"
leverage = 1.0

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
    
    # --- 4h Indicators (LTF) ---
    # Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper_band = highest_20
    lower_band = lowest_20
    
    # 4h volume confirmation: > 1.5x 20-period average (moderate to avoid overtrading)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_4h = volume > (1.5 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ADX(14) for trend strength
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
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        adx = np.zeros_like(high)
        
        atr[period] = np.nanmean(tr[1:period+1])
        plus_di[period] = np.nanmean(plus_dm[1:period+1]) / atr[period] * 100 if atr[period] != 0 else 0
        minus_di[period] = np.nanmean(minus_dm[1:period+1]) / atr[period] * 100 if atr[period] != 0 else 0
        
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_di[i] = (plus_di[i-1] * (period-1) + plus_dm[i]) / atr[i] * 100 if atr[i] != 0 else 0
            minus_di[i] = (minus_di[i-1] * (period-1) + minus_dm[i]) / atr[i] * 100 if atr[i] != 0 else 0
            dx[i] = abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) * 100 if (plus_di[i] + minus_di[i]) != 0 else 0
        
        adx[period:] = pd.Series(dx[period:]).ewm(span=period, adjust=False).mean().values
        return adx
    
    adx_14_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    adx_strong = adx_14_1d_aligned > 25  # Strong trend threshold
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if missing data
        if (np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or
            np.isnan(volume_confirm_4h[i]) or
            np.isnan(adx_strong[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper band + strong ADX trend + volume confirmation
            if (close[i] > upper_band[i] and 
                adx_strong[i] and 
                volume_confirm_4h[i]):
                signals[i] = 0.30
                position = 1
            # SHORT: Price breaks below lower band + strong ADX trend + volume confirmation
            elif (close[i] < lower_band[i] and 
                  adx_strong[i] and 
                  volume_confirm_4h[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below lower band
            if close[i] < lower_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price breaks above upper band
            if close[i] > upper_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals