#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h HMA trend filter and volume confirmation.
# Long when price breaks above Donchian upper AND 12h HMA rising AND volume > 1.5x 20-period average.
# Short when price breaks below Donchian lower AND 12h HMA falling AND volume > 1.5x 20-period average.
# Exit when price crosses Donchian midpoint (mean of upper/lower) or ATR-based stoploss (2*ATR from entry).
# Uses discrete position size 0.25. Designed to capture breakouts in trending markets with volume confirmation.
# Works in both bull and bear markets by requiring HMA trend filter, avoiding false breakouts in ranging markets.
# Target: 75-200 total trades over 4 years (19-50/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # === 12h Indicators: HMA (21-period) for trend ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Hull Moving Average: WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='full')[-len(values):] / weights.sum()
    
    half_len = 12 // 2
    sqrt_len = int(np.sqrt(12))
    
    wma_half = pd.Series(close_12h).rolling(window=half_len, min_periods=half_len).apply(lambda x: wma(x, half_len), raw=False).values
    wma_full = pd.Series(close_12h).rolling(window=12, min_periods=12).apply(lambda x: wma(x, 12), raw=False).values
    raw_hma = 2 * wma_half - wma_full
    hma_12h = pd.Series(raw_hma).rolling(window=sqrt_len, min_periods=sqrt_len).apply(lambda x: wma(x, sqrt_len), raw=False).values
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # HMA slope: rising if current > previous, falling if current < previous
    hma_slope = np.diff(hma_12h_aligned, prepend=hma_12h_aligned[0])
    hma_rising = hma_slope > 0
    hma_falling = hma_slope < 0
    
    # === Volume Confirmation: volume > 1.5x 20-period average ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # === 4h ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for HMA/ATR/Donchian)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(hma_12h_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(atr_4h[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_4h[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses Donchian midpoint (mean reversion)
            if price < donchian_mid[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses Donchian midpoint (mean reversion)
            if price > donchian_mid[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR above entry
            elif price > entry_price + 2.0 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian upper AND HMA rising AND volume spike
            if price > donchian_upper[i] and hma_rising[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Donchian lower AND HMA falling AND volume spike
            elif price < donchian_lower[i] and hma_falling[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Donchian20_12hHMATrend_VolumeSpike_V1"
timeframe = "4h"
leverage = 1.0