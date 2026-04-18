#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla H3/L3 breakout with weekly volatility filter and volume confirmation.
# Camarilla levels provide mean-reversion boundaries; breakouts beyond H3/L3 indicate strong momentum.
# Weekly ATR filter ensures we trade only when volatility is sufficient, avoiding choppy markets.
# Volume confirmation adds conviction to breakouts. Designed for low trade frequency (<30/year) to minimize fee drag.
# Works in bull markets (breakouts above H3) and bear markets (breakouts below L3).
name = "12h_Camarilla_H3L3_WeeklyATR_Volume_Filter"
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
    
    # Get weekly data for Camarilla levels and ATR filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Camarilla levels (H3, L3) from previous weekly bar
    # H3 = close + 1.1*(high - low)/2, L3 = close - 1.1*(high - low)/2
    typical_range = df_1w['high'] - df_1w['low']
    H3 = df_1w['close'] + 1.1 * typical_range / 2
    L3 = df_1w['close'] - 1.1 * typical_range / 2
    
    # Align weekly Camarilla levels to 12h timeframe (wait for weekly bar to close)
    H3_aligned = align_htf_to_ltf(prices, df_1w, H3.values)
    L3_aligned = align_htf_to_ltf(prices, df_1w, L3.values)
    
    # Calculate weekly ATR (14-period) for volatility filter
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # True Range calculation
    tr1 = high_w[1:] - low_w[1:]
    tr2 = np.abs(high_w[1:] - close_w[:-1])
    tr3 = np.abs(low_w[1:] - close_w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR using Wilder's smoothing (EMA with alpha=1/14)
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
    atr_mult = 2.0
    atr_threshold = atr * atr_mult
    
    # Align weekly ATR threshold to 12h timeframe
    atr_threshold_aligned = align_htf_to_ltf(prices, df_1w, atr_threshold)
    
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
            # Long: price breaks above H3 AND volume confirmation AND volatility filter
            long_breakout = close[i] > H3_aligned[i]
            if vol_confirm and vol_filter and long_breakout:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 AND volume confirmation AND volatility filter
            elif vol_confirm and vol_filter and close[i] < L3_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below L3 OR ATR drops below threshold (volatility collapse)
            exit_condition = close[i] < L3_aligned[i] or (np.isnan(atr_threshold_aligned[i]) or atr_threshold_aligned[i] <= 0)
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above H3 OR ATR drops below threshold (volatility collapse)
            exit_condition = close[i] > H3_aligned[i] or (np.isnan(atr_threshold_aligned[i]) or atr_threshold_aligned[i] <= 0)
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals