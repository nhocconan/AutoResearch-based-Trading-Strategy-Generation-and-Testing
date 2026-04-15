#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w HMA21 trend filter and volume confirmation
# Long when price breaks above Donchian upper (20-period high) + 1w HMA21 uptrend + volume > 2.0x 20-period avg
# Short when price breaks below Donchian lower (20-period low) + 1w HMA21 downtrend + volume > 2.0x 20-period avg
# Uses discrete position sizing (0.30) to balance return and drawdown.
# 1w HMA21 provides strong trend filter reducing whipsaws in both bull and bear markets.
# Volume threshold (2.0x) targets ~15-25 trades/year on 1d timeframe to avoid overtrading.
# Donchian channels provide clear structure-based entries that work in ranging and trending markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # === 1w Indicator: HMA21 ===
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    def hma(values, window):
        half = window // 2
        sqrt_n = int(np.sqrt(window))
        if half == 0:
            return np.full_like(values, np.nan)
        wma_half = wma(values, half)
        wma_full = wma(values, window)
        # Align arrays: wma_half starts at index half-1, wma_full starts at index window-1
        # We need to compute 2*wma_half - wma_full with proper alignment
        raw = 2 * wma_half - wma_full[half-1:]  # Shift wma_full to align with wma_half start
        return wma(raw, sqrt_n)
    
    close_1w = df_1w['close'].values
    hma_21_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 21:
        hma_21_1w[20:] = hma(close_1w, 21)
    hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w)
    
    # === 1d Donchian(20) ===
    # Upper = 20-period high, Lower = 20-period low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(34, 20) + 5  # HMA21 needs ~34 bars (20 for WMA + sqrt(20)~4 for final WMA), plus Donchian(20) + volume(20)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(hma_21_1w_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Donchian upper (close > upper)
        # 2. 1w HMA21 uptrend (close > HMA21)
        # 3. Volume confirmation
        if (close[i] > donchian_upper[i]) and \
           (close[i] > hma_21_1w_aligned[i]) and vol_confirm:
            signals[i] = 0.30
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Donchian lower (close < lower)
        # 2. 1w HMA21 downtrend (close < HMA21)
        # 3. Volume confirmation
        elif (close[i] < donchian_lower[i]) and \
             (close[i] < hma_21_1w_aligned[i]) and vol_confirm:
            signals[i] = -0.30
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1d_Donchian20_1wHMA21_Volume_Filter_v1"
timeframe = "1d"
leverage = 1.0