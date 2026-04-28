#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h TRIX(12) + 1d Volume Spike + 1w Choppiness Regime Filter
# TRIX(12) > 0 = bullish momentum, < 0 = bearish momentum
# Volume Spike = 1d volume > 2.0 x 20-bar average (confirms institutional interest)
# Choppiness Regime: 1w CHOP(14) > 61.8 = ranging (mean revert), < 38.2 = trending (follow momentum)
# Long: TRIX > 0 AND Volume Spike AND 1w CHOP < 38.2 (trending up)
# Short: TRIX < 0 AND Volume Spike AND 1w CHOP < 38.2 (trending down)
# Exit: TRIX crosses zero OR volume drops OR regime changes to ranging
# Target: 12-37 trades/year via strict confluence reducing false signals
# Works in bull/bear by only trading when 1w regime confirms trending conditions

name = "12h_TRIX12_1dVolumeSpike_1wChopRegime_v1"
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
    
    # Get 1d data for TRIX and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for TRIX and volume MA
        return np.zeros(n)
    
    # Get 1w data for Choppiness Index
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need sufficient data for CHOP
        return np.zeros(n)
    
    # Calculate TRIX(12) on 1d close
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) - 1 period ago, then / previous value * 100
    close_1d = df_1d['close'].values
    ema1 = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix_raw = (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1) * 100
    trix_raw[0] = 0  # First value has no previous
    
    # Calculate 1d volume and its 20-bar MA
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > 2.0 * volume_ma_20_1d
    
    # Calculate 1w Choppiness Index CHOP(14)
    # CHOP = 100 * log10(sum(TR14) / (ATR14 * 14)) / log10(14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Sum of TR14
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # ATR14 (average true range)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index
    chop_denominator = atr_14 * 14
    chop_ratio = np.where(chop_denominator > 0, tr_sum_14 / chop_denominator, 1.0)
    chop_ratio = np.where(np.isnan(chop_ratio), 1.0, chop_ratio)
    chop_1w = 100 * np.log10(chop_ratio) / np.log10(14)
    chop_1w = np.where(np.isnan(chop_1w), 50.0, chop_1w)  # Neutral if undefined
    
    # Prepend zeros for alignment (since we lost first bar in calculations)
    trix_raw = np.concatenate([np.full(35, np.nan), trix_raw])  # 12*3 + 1 - 1 = 35
    volume_spike_1d = np.concatenate([np.full(19, np.nan), volume_spike_1d])  # 20-1 = 19
    chop_1w = np.concatenate([np.full(27, np.nan), chop_1w])  # 13 (TR) + 14 (ATR) - 1 = 27
    
    # Align 1d indicators to 12h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix_raw)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Align 1w indicators to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 30)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        vol_spike = volume_spike_aligned[i]
        trix_val = trix_aligned[i]
        chop_val = chop_aligned[i]
        
        # Regime filter: only trade when 1w CHOP < 38.2 (trending market)
        is_trending = chop_val < 38.2
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when TRIX > 0 AND Volume Spike AND trending regime
            if trix_val > 0 and vol_spike and is_trending:
                signals[i] = 0.25
                position = 1
            # Short when TRIX < 0 AND Volume Spike AND trending regime
            elif trix_val < 0 and vol_spike and is_trending:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when TRIX <= 0 OR no volume spike OR ranging regime
            if trix_val <= 0 or not vol_spike or not is_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when TRIX >= 0 OR no volume spike OR ranging regime
            if trix_val >= 0 or not vol_spike or not is_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals