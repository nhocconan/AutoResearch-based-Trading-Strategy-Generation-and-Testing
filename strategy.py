#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Volume-Weighted Average Price (VWAP) Deviation + 1d ATR Regime Filter
# Uses 1d ATR(14) to filter regime: ATR percentile > 0.7 = high volatility (trade mean reversion),
# ATR percentile < 0.3 = low volatility (avoid). Price deviation from 12h VWAP acts as entry signal.
# Long when price < VWAP - 1.5 * ATR and ATR regime favors mean reversion.
# Short when price > VWAP + 1.5 * ATR and ATR regime favors mean reversion.
# Designed for low frequency (50-150 trades over 4 years) with clear structure in both bull and bear markets.

name = "12h_VWAP_ATR_Regime_MeanReversion_v1"
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
    
    # 1d HTF data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 12-period VWAP for 12h timeframe
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = vwap_numerator / vwap_denominator
    
    # 1d ATR(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # ATR using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    atr_period = 14
    atr = np.full_like(tr, np.nan)
    if len(tr) >= atr_period:
        # First ATR value is simple average of first 'atr_period' TR values
        first_atr = np.nanmean(tr[1:atr_period+1])
        atr[atr_period] = first_atr
        # Subsequent values: ATR[t] = (ATR[t-1] * (period-1) + TR[t]) / period
        for i in range(atr_period + 1, len(tr)):
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Calculate ATR percentile rank over 50-period lookback for regime classification
    atr_percentile = np.full_like(atr, np.nan)
    lookback = 50
    for i in range(lookback, len(atr)):
        window =atr[i-lookback:i+1]
        valid_window = window[~np.isnan(window)]
        if len(valid_window) >= 10:  # Minimum samples for meaningful percentile
            current_atr = atr[i]
            if not np.isnan(current_atr):
                percentile = (np.sum(valid_window <= current_atr) / len(valid_window)) * 100
                atr_percentile[i] = percentile
    
    # Align ATR percentile to 12h timeframe
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(lookback, 20)  # Need ATR percentile and VWAP
    
    for i in range(start_idx, n):
        if (np.isnan(vwap[i]) or np.isnan(atr_percentile_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filters based on ATR percentile
        high_vol_regime = atr_percentile_aligned[i] > 70  # ATR > 70th percentile = high volatility
        low_vol_regime = atr_percentile_aligned[i] < 30   # ATR < 30th percentile = low volatility
        mean_reversion_favorable = high_vol_regime  # Trade mean reversion in high volatility
        
        if position == 0:  # Flat - look for new entries
            # Only trade in high volatility regime where mean reversion is expected
            if mean_reversion_favorable:
                # Long: Price significantly below VWAP
                if close[i] < vwap[i] - (1.5 * atr[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: Price significantly above VWAP
                elif close[i] > vwap[i] + (1.5 * atr[i]):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid low volatility and transition regimes
        
        elif position == 1:  # Long position
            # Exit conditions: price returns to VWAP or opposite extreme
            exit_long = False
            if close[i] >= vwap[i]:  # Return to VWAP
                exit_long = True
            elif close[i] > vwap[i] + (1.5 * atr[i]):  # Opposite extreme
                exit_long = True
            
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: price returns to VWAP or opposite extreme
            exit_short = False
            if close[i] <= vwap[i]:  # Return to VWAP
                exit_short = True
            elif close[i] < vwap[i] - (1.5 * atr[i]):  # Opposite extreme
                exit_short = True
            
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals