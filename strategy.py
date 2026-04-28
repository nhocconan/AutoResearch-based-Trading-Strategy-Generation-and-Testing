#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h TRIX crossover with 1d volume spike filter and ADX regime filter
# Long when TRIX crosses above zero AND 1d volume > 2.0x 20-bar avg AND ADX < 25 (range regime)
# Short when TRIX crosses below zero AND 1d volume > 2.0x 20-bar avg AND ADX < 25 (range regime)
# Exit when TRIX crosses zero in opposite direction
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year on 12h.
# Works in bull markets by capturing momentum during low-volatility breakouts
# Works in bear markets by requiring volume spikes which often accompany panic selling/buying climaxes that precede reversals
# ADX < 25 filter ensures we only trade in ranging markets where mean reversion works best

name = "12h_TRIX_ZeroCross_1dVolumeSpike_ADXRegime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume spike and ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need sufficient data for volume MA and ADX
        return np.zeros(n)
    
    # Calculate TRIX on 12h close (triple EMA)
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = 100 * (ema3.pct_change())
    trix_values = trix.values
    
    # Calculate 1d volume spike filter
    volume_1d = df_1d['volume'].values
    volume_series_1d = pd.Series(volume_1d)
    volume_ma_20_1d = volume_series_1d.rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > 2.0 * volume_ma_20_1d
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate ADX on 1d timeframe for regime filter
    # ADX calculation: +DM, -DM, TR, then smoothed averages
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # +DM and -DM
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed averages (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(values[1:period])  # Skip first NaN
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    period = 14
    atr_1d = wilders_smoothing(tr, period)
    plus_di_1d = 100 * wilders_smoothing(plus_dm, period) / atr_1d
    minus_di_1d = 100 * wilders_smoothing(minus_dm, period) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilders_smoothing(dx_1d, period)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Regime filter: ADX < 25 indicates ranging market (good for mean reversion)
    ranging_regime = adx_1d_aligned < 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(45, 20)  # Need sufficient history for TRIX and ADX
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix_values[i]) or np.isnan(volume_spike_1d_aligned[i]) or 
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        vol_spike = volume_spike_1d_aligned[i]
        ranging = ranging_regime[i]
        curr_trix = trix_values[i]
        prev_trix = trix_values[i-1]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when TRIX crosses above zero AND volume spike AND ranging regime
            if prev_trix <= 0 and curr_trix > 0 and vol_spike and ranging:
                signals[i] = 0.25
                position = 1
            # Short when TRIX crosses below zero AND volume spike AND ranging regime
            elif prev_trix >= 0 and curr_trix < 0 and vol_spike and ranging:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when TRIX crosses below zero
            if prev_trix >= 0 and curr_trix < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when TRIX crosses above zero
            if prev_trix <= 0 and curr_trix > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals