#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR volatility filter and volume confirmation
# Long when price breaks above 4h Donchian upper + current ATR(14) > 1d ATR(14) average + volume > 1.5x 20-period avg
# Short when price breaks below 4h Donchian lower + current ATR(14) > 1d ATR(14) average + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (20-40/year).
# ATR filter ensures we only trade during elevated volatility, avoiding low-volume chop.
# Works in bull markets (trend continuation) and bear markets (strong downtrends) by requiring volatility expansion.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicator: ATR (volatility filter) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    high_1d_shift = np.roll(high_1d, 1)
    low_1d_shift = np.roll(low_1d, 1)
    close_1d_shift = np.roll(close_1d, 1)
    high_1d_shift[0] = high_1d[0]
    low_1d_shift[0] = low_1d[0]
    close_1d_shift[0] = close_1d[0]
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_1d_shift)
    tr3 = np.abs(low_1d - close_1d_shift)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing for ATR
    period = 14
    atr_1d = np.zeros_like(tr)
    atr_1d[period-1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * (period-1) + tr[i]) / period
    
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 4h Indicators: Donchian Channel (20-period) and ATR ===
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # 4h ATR for volatility comparison
    high_4h_shift = np.roll(high, 1)
    low_4h_shift = np.roll(low, 1)
    close_4h_shift = np.roll(close, 1)
    high_4h_shift[0] = high[0]
    low_4h_shift[0] = low[0]
    close_4h_shift[0] = close[0]
    
    tr1_4h = high - low
    tr2_4h = np.abs(high - close_4h_shift)
    tr3_4h = np.abs(low - close_4h_shift)
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    
    atr_4h = np.zeros_like(tr_4h)
    atr_4h[period-1] = np.mean(tr_4h[:period])
    for i in range(period, len(tr_4h)):
        atr_4h[i] = (atr_4h[i-1] * (period-1) + tr_4h[i]) / period
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(donchian_window, period) + 20  # Donchian(20) + ATR(14) + volume(20)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(atr_4h[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current 4h ATR > 1d ATR average (expanded volatility)
        vol_filter = atr_4h[i] > atr_1d_aligned[i]
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 4h Donchian upper (20-period)
        # 2. Volatility expansion (current ATR > 1d ATR)
        # 3. Volume confirmation
        if (close[i] > donchian_high[i]) and \
           vol_filter and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 4h Donchian lower (20-period)
        # 2. Volatility expansion (current ATR > 1d ATR)
        # 3. Volume confirmation
        elif (close[i] < donchian_low[i]) and \
             vol_filter and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_1dATR_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0