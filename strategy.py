#!/usr/bin/env python3
"""
6h Williams %R Mean Reversion with 12h Trend Filter and Volume Confirmation
Hypothesis: Williams %R (14) identifies overbought/oversold conditions on 6h timeframe.
In ranging markets (Choppiness Index > 61.8 from 12h), we mean-revert at extremes:
- Long when Williams %R < -80 (oversold) AND 12h EMA50 trend is up (close > EMA) AND volume > 1.5x average
- Short when Williams %R > -20 (overbought) AND 12h EMA50 trend is down (close < EMA) AND volume > 1.5x average
Exit when Williams %R crosses -50 (mean reversion complete) or ATR trailing stop (2.5*ATR).
Uses discrete position sizing (0.25) targeting ~15-35 trades/year on 6h timeframe.
Combines momentum oscillator mean reversion with trend filter and volume confirmation for robustness.
Williams %R calculated from prior completed 6h bars, ensuring no look-ahead bias.
Choppiness Index uses 12h data to determine regime (range vs trend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12h Choppiness Index for regime filter (range > 61.8, trend < 38.2)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # True Range for 12h
    tr1_12h = np.abs(high_12h - low_12h)
    tr2_12h = np.abs(high_12h - np.roll(close_12h, 1))
    tr3_12h = np.abs(low_12h - np.roll(close_12h, 1))
    tr1_12h[0] = 0
    tr2_12h[0] = 0
    tr3_12h[0] = 0
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods for Chop calculation
    hh_12h = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    ll_12h = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    sum_tr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).sum().values
    chop_denominator = hh_12h - ll_12h
    # Avoid division by zero
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)
    chop_12h = 100 * (np.log10(sum_tr_12h) - np.log10(chop_denominator)) / np.log10(14)
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # Williams %R (14) on 6h: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Values: 0 to -100, where > -20 is overbought, < -80 is oversold
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    hh_ll_diff = highest_high - lowest_low
    hh_ll_diff = np.where(hh_ll_diff == 0, 1e-10, hh_ll_diff)
    williams_r = (highest_high - close) / hh_ll_diff * -100
    
    # 6h volume average (20-period) for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for 6h trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 14)  # vol MA20, EMA50, Williams %R14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(chop_12h_aligned[i]) or
            np.isnan(williams_r[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema_val = ema_50_12h_aligned[i]
        chop_val = chop_12h_aligned[i]
        wr = williams_r[i]
        
        if position == 0:
            # Only trade in ranging markets (Chop > 61.8)
            if chop_val > 61.8:
                # Long: Williams %R oversold (< -80) AND bullish 12h trend (close > EMA50) AND volume spike
                if wr < -80.0 and close[i] > ema_val and volume[i] > 1.5 * vol_ma_val:
                    signals[i] = 0.25
                    position = 1
                    highest_since_entry = price
                # Short: Williams %R overbought (> -20) AND bearish 12h trend (close < EMA50) AND volume spike
                elif wr > -20.0 and close[i] < ema_val and volume[i] > 1.5 * vol_ma_val:
                    signals[i] = -0.25
                    position = -1
                    lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Williams %R crosses -50 (mean reversion complete)
            if position == 1 and wr > -50.0:
                exit_signal = True
            elif position == -1 and wr < -50.0:
                exit_signal = True
            
            # ATR-based trailing stop: 2.5 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_MeanReversion_12hEMA50_Trend_VolumeSpike_ChopRegime_WR50Exit_ATRTrailingStop"
timeframe = "6h"
leverage = 1.0