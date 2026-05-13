#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray Power (Bull/Bear) with 1d ADX regime filter and volume confirmation.
# Elder Ray Bull Power = High - EMA13; Bear Power = EMA13 - Low.
# Long when Bull Power > 0 AND Bear Power < 0 (strong bullish momentum) with 1d ADX > 25 (trending) and volume > 1.5x MA20.
# Short when Bull Power < 0 AND Bear Power > 0 (strong bearish momentum) with 1d ADX > 25 and volume > 1.5x MA20.
# Exit when momentum weakens: Bull Power <= 0 for longs or Bear Power <= 0 for shorts.
# Uses discrete position sizing (0.25) to minimize fee drag and manage drawdown.
# Designed for low trade frequency (~12-37/year) to work in both bull and bear markets by requiring strong momentum, trend alignment, and volume confirmation.

name = "6h_ElderRay_Power_1dADX_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14) on 1d data
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
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * (np.zeros_like(plus_dm))
        minus_di = 100 * (np.zeros_like(minus_dm))
        plus_smoothed = np.zeros_like(plus_dm)
        minus_smoothed = np.zeros_like(minus_dm)
        
        plus_smoothed[period] = np.mean(plus_dm[1:period+1])
        minus_smoothed[period] = np.mean(minus_dm[1:period+1])
        
        for i in range(period+1, len(plus_dm)):
            plus_smoothed[i] = (plus_smoothed[i-1] * (period-1) + plus_dm[i]) / period
            minus_smoothed[i] = (minus_smoothed[i-1] * (period-1) + minus_dm[i]) / period
        
        plus_di = 100 * plus_smoothed / (atr + 1e-10)
        minus_di = 100 * minus_smoothed / (atr + 1e-10)
        
        dx = np.zeros_like(close)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        
        adx = np.zeros_like(close)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(adx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Elder Ray Power on 6d data
    # Bull Power = High - EMA13(Close)
    # Bear Power = EMA13(Close) - Low
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(adx_1d_aligned[i]) or np.isnan(ema13[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Strong bullish momentum (Bull Power > 0 AND Bear Power < 0) with 1d ADX > 25 and volume spike
            if bull_power[i] > 0 and bear_power[i] < 0 and adx_1d_aligned[i] > 25 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Strong bearish momentum (Bull Power < 0 AND Bear Power > 0) with 1d ADX > 25 and volume spike
            elif bull_power[i] < 0 and bear_power[i] > 0 and adx_1d_aligned[i] > 25 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: When bullish momentum weakens (Bull Power <= 0)
            if bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: When bearish momentum weakens (Bear Power <= 0)
            if bear_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals