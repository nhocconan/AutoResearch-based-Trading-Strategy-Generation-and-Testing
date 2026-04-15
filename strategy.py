#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation
# Long when price breaks above 4h Donchian upper (20-period) + 1d ATR(14) > 20-period SMA of ATR + volume > 1.5x 20-period avg
# Short when price breaks below 4h Donchian lower (20-period) + 1d ATR(14) > 20-period SMA of ATR + volume > 1.5x 20-period avg
# Uses ATR regime to filter choppy markets (low ATR) and only trade when volatility is expanding.
# Designed for low trade frequency (20-40/year) with discrete position sizing (0.25) to minimize fee churn.

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
    
    # === 1d Indicator: ATR (volatility regime filter) ===
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
    
    # Wilder's ATR (14-period)
    period = 14
    atr_1d = np.zeros_like(tr)
    atr_1d[period-1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * (period-1) + tr[i]) / period
    
    # 20-period SMA of ATR for regime comparison
    atr_sma_20 = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_regime = atr_1d > atr_sma_20  # Volatility expanding when ATR > its SMA
    
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime)
    
    # === 4h Indicator: Donchian Channel (20-period) ===
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(donchian_window, 20) + 20  # Donchian(20) + ATR(14+20) + volume(20)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr_regime_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 4h Donchian upper (20-period)
        # 2. Volatility regime: 1d ATR > 20-period SMA of ATR (expanding volatility)
        # 3. Volume confirmation
        if (close[i] > donchian_high[i]) and \
           atr_regime_aligned[i] and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 4h Donchian lower (20-period)
        # 2. Volatility regime: 1d ATR > 20-period SMA of ATR (expanding volatility)
        # 3. Volume confirmation
        elif (close[i] < donchian_low[i]) and \
             atr_regime_aligned[i] and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_1dATR_Regime_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0