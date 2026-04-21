#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index (14) regime filter + 4h TRIX (12,20,9) crossover with volume spike.
# Uses TRIX for momentum in trending regimes and mean reversion in choppy regimes.
# Works in both bull and bear markets by adapting to market regime.
# Target: 20-40 trades/year by requiring regime alignment and volume confirmation.
# Entry: Long when TRIX crosses above signal AND CHOP > 61.8 (choppy) with volume spike.
#        Short when TRIX crosses below signal AND CHOP > 61.8 (choppy) with volume spike.
#        In trending regime (CHOP < 38.2), follow TRIX crossovers without volume filter.
# Exit: Opposite TRIX crossover or regime change.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for TRIX and Choppiness calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate TRIX (12,20,9) on daily close
    close = df_1d['close'].values
    # First EMA 12
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    # Second EMA 20 of first EMA
    ema2 = pd.Series(ema1).ewm(span=20, adjust=False, min_periods=20).mean().values
    # Third EMA 20 of second EMA
    ema3 = pd.Series(ema2).ewm(span=20, adjust=False, min_periods=20).mean().values
    # TRIX = 100 * (ema3_today - ema3_yesterday) / ema3_yesterday
    trix_raw = 100 * (ema3[1:] - ema3[:-1]) / ema3[:-1]
    trix = np.concatenate([np.array([np.nan]), trix_raw])
    # TRIX signal line (9-period EMA of TRIX)
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Calculate Choppiness Index (14) on daily high/low/close
    high = df_1d['high'].values
    low = df_1d['low'].values
    close_d = df_1d['close'].values
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close_d[:-1])
    tr3 = np.abs(low[1:] - close_d[:-1])
    tr = np.concatenate([np.array([np.nan]), np.maximum(tr1, np.maximum(tr2, tr3))])
    # ATR(14) of TR
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    # Sum of ATR over 14 periods
    sum_atr14 = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Chop = 100 * log10(sum(ATR14) / (HH - LL)) / log10(14)
    # Avoid division by zero
    range_hl = hh - ll
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    chop = 100 * np.log10(sum_atr14 / range_hl) / np.log10(14)
    
    # Volume confirmation: daily volume > 1.5x 20-day average
    volume = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * vol_ma_20
    
    # Align daily indicators to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    trix_signal_aligned = align_htf_to_ltf(prices, df_1d, trix_signal)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(trix_aligned[i]) or np.isnan(trix_signal_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_confirm_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        trix_val = trix_aligned[i]
        trix_sig_val = trix_signal_aligned[i]
        chop_val = chop_aligned[i]
        vol_confirm = volume_confirm_aligned[i]
        
        # TRIX crossover signals
        trix_cross_up = trix_val > trix_sig_val and (i == 50 or trix_aligned[i-1] <= trix_signal_aligned[i-1])
        trix_cross_down = trix_val < trix_sig_val and (i == 50 or trix_aligned[i-1] >= trix_signal_aligned[i-1])
        
        # Regime filters
        choppy = chop_val > 61.8  # Choppy regime - mean reversion
        trending = chop_val < 38.2  # Trending regime - trend follow
        
        if position == 0:
            # Enter long in choppy regime with TRIX cross up and volume confirmation
            if choppy and trix_cross_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short in choppy regime with TRIX cross down and volume confirmation
            elif choppy and trix_cross_down and vol_confirm:
                signals[i] = -0.25
                position = -1
            # Enter long in trending regime with TRIX cross up (no volume filter needed)
            elif trending and trix_cross_up:
                signals[i] = 0.25
                position = 1
            # Enter short in trending regime with TRIX cross down (no volume filter needed)
            elif trending and trix_cross_down:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: TRIX cross down or regime change to choppy without confirmation
                if trix_cross_down:
                    exit_signal = True
                elif choppy and not vol_confirm:  # In choppy, need volume to hold
                    exit_signal = True
            elif position == -1:
                # Exit short: TRIX cross up or regime change to choppy without confirmation
                if trix_cross_up:
                    exit_signal = True
                elif choppy and not vol_confirm:  # In choppy, need volume to hold
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_TRIX_Chop_Regime_Volume"
timeframe = "4h"
leverage = 1.0