#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR volatility filter and volume confirmation
# Long when price breaks above 4h Donchian upper (20-period) + 1d ATR ratio > 0.8 (sufficient volatility) + volume > 1.3x 20-period avg
# Short when price breaks below 4h Donchian lower (20-period) + 1d ATR ratio > 0.8 + volume > 1.3x 20-period avg
# ATR ratio = current ATR(14) / 50-period SMA of ATR - ensures we trade when volatility is elevated but not extreme
# Designed for low trade frequency (15-25/year) with discrete sizing (0.25) to minimize fee churn.
# Works in bull markets (trend continuation) and bear markets (volatility expansion on breakdowns).

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
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d Indicator: ATR and its SMA for volatility regime filter ===
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
    
    # ATR(14) using Wilder's smoothing
    period_atr = 14
    atr_1d = np.zeros_like(tr)
    atr_1d[period_atr-1] = np.mean(tr[:period_atr])
    for i in range(period_atr, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * (period_atr-1) + tr[i]) / period_atr
    
    # 50-period SMA of ATR for volatility regime
    atr_sma_50 = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    # ATR ratio: current ATR / ATR SMA - values > 1 indicate elevated volatility
    atr_ratio = np.where(atr_sma_50 > 0, atr_1d / atr_sma_50, 0)
    
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # === 4h Indicator: Donchian Channel (20-period) ===
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(donchian_window, 50) + 20  # Donchian(20) + ATR_SMA(50) + volume(20)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.3)
        
        # Volatility filter: ATR ratio between 0.8 and 3.0 (avoid low vol chop and extreme volatility)
        vol_filter = (atr_ratio_aligned[i] >= 0.8) and (atr_ratio_aligned[i] <= 3.0)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 4h Donchian upper (20-period)
        # 2. Sufficient volatility for meaningful moves
        # 3. Volume confirmation
        if (close[i] > donchian_high[i]) and \
           vol_filter and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 4h Donchian lower (20-period)
        # 2. Sufficient volatility for meaningful moves
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