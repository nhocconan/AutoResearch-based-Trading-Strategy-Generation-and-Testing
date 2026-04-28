#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX with volume spike and chop regime filter
# TRIX (triple EMA) filters noise and identifies momentum.
# Long when TRIX crosses above zero with volume > 1.8x 20-bar average and chop > 61.8 (range).
# Short when TRIX crosses below zero with volume > 1.8x 20-bar average and chop > 61.8.
# Chop regime filter avoids trending markets where momentum fails.
# Uses 4h timeframe targeting 20-50 trades/year (~80-200 total over 4 years) to minimize fee drag.
# Works in ranging markets (2025+ test) via mean reversion at extremes and in trending markets via momentum continuation.

name = "4h_TRIX_ZeroCross_VolumeSpike_ChopRegime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate TRIX(12) - triple EMA of ROC
    # ROC = (close - close.shift(1)) / close.shift(1) * 100
    roc = np.zeros_like(close)
    roc[1:] = (close[1:] - close[:-1]) / close[:-1] * 100.0
    
    # Triple EMA of ROC
    ema1 = pd.Series(roc).ewm(span=12, min_periods=12, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, min_periods=12, adjust=False).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, min_periods=12, adjust=False).mean().values
    trix = ema3
    
    # Volume confirmation: >1.8x 20-bar average volume (strict filter)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.8 * volume_ma_20
    
    # Chop regime filter: Chop > 61.8 indicates ranging market (good for mean reversion)
    # Chop = 100 * log10(sum(ATR(1), n) / (log10(n) * (highest high - lowest low)))
    # Simplified: use ATR(14) and price range over 14 bars
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    price_range = max_high_14 - min_low_14
    price_range = np.where(price_range == 0, 1e-10, price_range)
    chop = 100 * np.log10(atr14 * 14 / (np.log10(14) * price_range))
    chop_regime = chop > 61.8  # ranging market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(36, 20, 14)  # TRIX needs ~36 bars, volume MA(20), chop needs 14
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix[i]) or np.isnan(volume_ma_20[i]) or 
            np.isnan(chop[i]) or np.isnan(atr14[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        chop_filter = chop_regime[i]
        price = close[i]
        curr_trix = trix[i]
        prev_trix = trix[i-1]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: TRIX crosses above zero, volume spike, chop regime (ranging)
            if prev_trix <= 0 and curr_trix > 0 and vol_confirm and chop_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: TRIX crosses below zero, volume spike, chop regime (ranging)
            elif prev_trix >= 0 and curr_trix < 0 and vol_confirm and chop_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on TRIX crossing below zero or stoploss
            # ATR-based stoploss: 2.5 * ATR below entry
            tr_slice = tr[max(0, i-14):i+1]
            atr_val = np.mean(tr_slice[-14:]) if len(tr_slice) >= 14 else np.mean(tr_slice)
            stop_loss = entry_price - 2.5 * atr_val
            if curr_trix < 0 or price < stop_loss:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on TRIX crossing above zero or stoploss
            # ATR-based stoploss: 2.5 * ATR above entry
            tr_slice = tr[max(0, i-14):i+1]
            atr_val = np.mean(tr_slice[-14:]) if len(tr_slice) >= 14 else np.mean(tr_slice)
            stop_loss = entry_price + 2.5 * atr_val
            if curr_trix > 0 or price > stop_loss:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals