#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX + volume spike + choppiness regime filter
# Long when TRIX crosses above zero AND chop > 61.8 (ranging) AND volume > 1.5x 20-period average
# Short when TRIX crosses below zero AND chop > 61.8 (ranging) AND volume > 1.5x 20-period average
# Uses ATR-based trailing stop (2.0x ATR) for risk management
# Discrete position sizing (0.25) to minimize fee drag
# Target: 25-35 trades/year on 4h timeframe (~100-140 total over 4 years)
# TRIX is effective in ranging markets (chop > 61.8) which suits 2025 bear/range conditions
# Volume confirmation reduces false signals
# Works in both bull and bear via mean reversion in ranging regimes

name = "4h_TRIX_ZeroCross_ChopRegime_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter (used as regime filter: price > EMA34 = bull, < EMA34 = bear)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate TRIX (15,9,9) - triple exponential moving average
    # TRIX = EMA(EMA(EMA(close, 15), 9), 9)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema3 = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean().values
    trix = np.where(ema3[:-1] != 0, (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100, 0)
    trix = np.concatenate([[0], trix])  # align length
    
    # Calculate TRIX signal line (9-period EMA of TRIX)
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Calculate TRIX histogram (TRIX - signal)
    trix_hist = trix - trix_signal
    
    # Calculate Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR(1)) / (nperiod * log(nperiod))) / log10(nperiod)
    # Simplified: CHOP = 100 * log10(sum(TR(1)) / (ATR(14) * 14)) / log10(14)
    atr_1 = tr  # true range
    sum_tr_14 = np.zeros(n)
    for i in range(14, n):
        sum_tr_14[i] = np.sum(atr_1[i-13:i+1])
    sum_tr_14[:14] = np.nan
    
    atr_14 = atr  # already calculated
    chop = np.zeros(n)
    for i in range(14, n):
        if atr_14[i] > 0 and sum_tr_14[i] > 0:
            chop[i] = 100 * np.log10(sum_tr_14[i] / (atr_14[i] * 14)) / np.log10(14)
        else:
            chop[i] = 50  # neutral
    chop[:14] = 50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    start_idx = max(100, 34, 14)  # warmup
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_trix = trix[i]
        curr_trix_signal = trix_signal[i]
        curr_trix_hist = trix_hist[i]
        curr_chop = chop[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        
        # Volume spike confirmation: current volume > 1.5x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 1.5 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            # Trailing stop: 2.0 * ATR below highest high
            stop_price = highest_high_since_entry - 2.0 * curr_atr
            # Exit conditions: price below trailing stop OR TRIX crosses below zero
            if curr_close < stop_price or curr_trix_hist < 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            # Trailing stop: 2.0 * ATR above lowest low
            stop_price = lowest_low_since_entry + 2.0 * curr_atr
            # Exit conditions: price above trailing stop OR TRIX crosses above zero
            if curr_close > stop_price or curr_trix_hist > 0:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: TRIX crosses above zero AND chop > 61.8 (ranging) AND volume spike
            # In ranging markets, we expect mean reversion - long when TRIX turns up from negative
            if curr_trix_hist > 0 and curr_trix_hist <= 0.1 and curr_chop > 61.8 and vol_spike:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = curr_high
            # Short entry: TRIX crosses below zero AND chop > 61.8 (ranging) AND volume spike
            # In ranging markets, we expect mean reversion - short when TRIX turns down from positive
            elif curr_trix_hist < 0 and curr_trix_hist >= -0.1 and curr_chop > 61.8 and vol_spike:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals