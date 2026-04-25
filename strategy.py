#!/usr/bin/env python3
"""
4h TRIX + Volume Spike + Choppiness Regime Filter
Hypothesis: TRIX (Triple Exponential Average) identifies momentum shifts with less lag than MACD.
Combined with volume confirmation to avoid false breakouts and choppiness regime filter (CHOP > 61.8 for mean reversion,
CHOP < 38.2 for trend following) to adapt to market conditions. Uses 1d EMA50 for higher-timeframe trend alignment.
Designed for low trade frequency (target: 20-50/year) to minimize fee drag while capturing strong moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average with min_periods"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for EMA50 trend (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Daily EMA50 trend filter
    ema_50_1d = calculate_ema(df_1d['close'].values, 50)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # TRIX calculation (15-period)
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15)
    ema1 = calculate_ema(close, 15)
    ema2 = calculate_ema(ema1, 15)
    ema3 = calculate_ema(ema2, 15)
    # Avoid division by zero
    ema3_prev = np.roll(ema3, 1)
    ema3_prev[0] = ema3[0] if not np.isnan(ema3[0]) else 0
    trix = 100 * (ema3 - ema3_prev) / ema3_prev
    trix[0] = 0
    
    # TRIX signal line (9-period EMA of TRIX)
    trix_signal = calculate_ema(trix, 9)
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    # Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR(1)) / (n * log(n))) / log10(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr1 = tr  # ATR(1) is just true range
    atr_sum = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    n_val = 14
    log_n = np.log10(n_val)
    chop = 100 * (np.log10(atr_sum) - log_n * np.log10(n_val)) / log_n
    # Handle edge cases
    chop = np.where((atr_sum > 0) & (n_val > 1), chop, 50.0)  # default to 50 (neutral)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for TRIX, EMA, volume MA, CHOP
    start_idx = max(15*3 + 9, 50, 20, 14) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix[i]) or np.isnan(trix_signal[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Regime-based logic
        # CHOP > 61.8 = ranging market (mean reversion)
        # CHOP < 38.2 = trending market (trend following)
        is_ranging = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        
        if position == 0:
            # Look for entry signals
            if is_ranging:
                # In ranging market: mean reversion at extremes
                # Long when TRIX crosses above signal AND TRIX is negative (oversold)
                # Short when TRIX crosses below signal AND TRIX is positive (overbought)
                trix_cross_above = trix[i] > trix_signal[i] and trix[i-1] <= trix_signal[i-1]
                trix_cross_below = trix[i] < trix_signal[i] and trix[i-1] >= trix_signal[i-1]
                
                long_entry = trix_cross_above and (trix[i] < 0) and volume_spike[i]
                short_entry = trix_cross_below and (trix[i] > 0) and volume_spike[i]
            else:  # trending or neutral
                # In trending market: follow TRIX momentum
                # Long when TRIX crosses above signal with volume spike
                # Short when TRIX crosses below signal with volume spike
                trix_cross_above = trix[i] > trix_signal[i] and trix[i-1] <= trix_signal[i-1]
                trix_cross_below = trix[i] < trix_signal[i] and trix[i-1] >= trix_signal[i-1]
                
                long_entry = trix_cross_above and volume_spike[i] and (close[i] > ema_50_1d_aligned[i])
                short_entry = trix_cross_below and volume_spike[i] and (close[i] < ema_50_1d_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when TRIX crosses below signal OR trend change
            trix_cross_below = trix[i] < trix_signal[i] and trix[i-1] >= trix_signal[i-1]
            trend_change = close[i] < ema_50_1d_aligned[i]
            
            if trix_cross_below or trend_change:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when TRIX crosses above signal OR trend change
            trix_cross_above = trix[i] > trix_signal[i] and trix[i-1] <= trix_signal[i-1]
            trend_change = close[i] > ema_50_1d_aligned[i]
            
            if trix_cross_above or trend_change:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_TRIX_VolumeSpike_ChopRegime_1dEMA50_Trend"
timeframe = "4h"
leverage = 1.0