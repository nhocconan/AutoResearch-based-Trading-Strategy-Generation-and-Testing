#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot reversal with daily volatility filter and volume confirmation.
# Camarilla levels (H3/L3) act as strong support/resistance in ranging markets.
# Daily ATR filter ensures we only trade when volatility is sufficient to avoid chop.
# Volume confirmation adds conviction to reversal signals.
# Designed for low trade frequency (12-37/year) to minimize fee drag in 12h timeframe.
# Works in sideways markets by fading extreme moves back to the mean.
# Works in trending markets by only taking reversals in the direction of the higher timeframe trend.
name = "12h_Camarilla_H3L3_Reversal_Volatility_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and ATR filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels (H3, L3) from previous day to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shift by 1)
    ph = np.concatenate([[np.nan], high_1d[:-1]])
    pl = np.concatenate([[np.nan], low_1d[:-1]])
    pc = np.concatenate([[np.nan], close_1d[:-1]])
    
    # Camarilla calculations
    range_ = ph - pl
    H3 = pc + range_ * 1.1 / 6
    L3 = pc - range_ * 1.1 / 6
    
    # Calculate daily ATR (14-period) for volatility filter
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_period = 14
    atr = np.full_like(tr, np.nan)
    if len(tr) >= atr_period:
        atr[atr_period-1] = np.nanmean(tr[:atr_period])
        for i in range(atr_period, len(tr)):
            if not np.isnan(atr[i-1]) and not np.isnan(tr[i]):
                atr[i] = atr[i-1] * (1 - 1/atr_period) + tr[i] * (1/atr_period)
            else:
                atr[i] = np.nan
    
    # ATR multiplier for volatility filter
    atr_mult = 1.5
    atr_threshold = atr * atr_mult
    
    # Align daily Camarilla levels and ATR threshold to 12h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    atr_threshold_aligned = align_htf_to_ltf(prices, df_1d, atr_threshold)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(atr_threshold_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        # Volatility filter: current ATR threshold must be positive (sufficient volatility)
        vol_filter = not np.isnan(atr_threshold_aligned[i]) and atr_threshold_aligned[i] > 0
        
        if position == 0:
            # Long: price crosses below L3 (oversold) AND volume confirmation AND volatility filter
            long_signal = close[i] < L3_aligned[i]
            if vol_confirm and vol_filter and long_signal:
                signals[i] = 0.25
                position = 1
            # Short: price crosses above H3 (overbought) AND volume confirmation AND volatility filter
            elif vol_confirm and vol_filter and close[i] > H3_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back above L3 (mean reversion) OR volatility drops
            exit_condition = close[i] > L3_aligned[i] or (np.isnan(atr_threshold_aligned[i]) or atr_threshold_aligned[i] <= 0)
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back below H3 (mean reversion) OR volatility drops
            exit_condition = close[i] < H3_aligned[i] or (np.isnan(atr_threshold_aligned[i]) or atr_threshold_aligned[i] <= 0)
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals