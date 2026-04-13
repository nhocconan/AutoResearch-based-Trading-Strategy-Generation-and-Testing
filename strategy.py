#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Bands squeeze + RSI mean reversion + 1d trend filter.
# Bollinger Band squeeze identifies low volatility periods that precede breakouts.
# RSI < 30 or > 70 with 1d EMA trend filter captures mean reversion in trending markets.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
# Target: 20-50 trades per year (80-200 total over 4 years) for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # EMA(50) for 1d trend filter
    ema50_1d = np.zeros(len(close_1d))
    ema_multiplier = 2 / (50 + 1)
    ema50_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema50_1d[i] = (close_1d[i] - ema50_1d[i-1]) * ema_multiplier + ema50_1d[i-1]
    
    # Align 1d EMA to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Bollinger Bands (20, 2) on 4h
    bb_length = 20
    bb_mult = 2.0
    basis = np.full(n, np.nan)
    dev = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(bb_length - 1, n):
        basis[i] = np.mean(close[i - bb_length + 1:i + 1])
        dev[i] = bb_mult * np.std(close[i - bb_length + 1:i + 1])
        upper[i] = basis[i] + dev[i]
        lower[i] = basis[i] - dev[i]
    
    # Bollinger Band width (normalized)
    bb_width = np.full(n, np.nan)
    for i in range(bb_length - 1, n):
        if basis[i] != 0:
            bb_width[i] = (upper[i] - lower[i]) / basis[i]
    
    # RSI (14) on 4h
    rsi_length = 14
    rsi = np.full(n, np.nan)
    change = np.zeros(n)
    change[1:] = close[1:] - close[:-1]
    gain = np.where(change > 0, change, 0)
    loss = np.where(change < 0, -change, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(rsi_length, n):
        if i == rsi_length:
            avg_gain[i] = np.mean(gain[1:rsi_length + 1])
            avg_loss[i] = np.mean(loss[1:rsi_length + 1])
        else:
            avg_gain[i] = (avg_gain[i-1] * (rsi_length - 1) + gain[i]) / rsi_length
            avg_loss[i] = (avg_loss[i-1] * (rsi_length - 1) + loss[i]) / rsi_length
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
    # Volume average (20-period)
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(max(bb_length, rsi_length, 20), n):
        # Skip if any required data is not ready
        if (np.isnan(bb_width[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        bbw = bb_width[i]
        rsi_val = rsi[i]
        ema_trend = ema50_1d_aligned[i]
        
        # Bollinger Band squeeze: low volatility (bottom 20% of BB width)
        # Calculate percentile of BB width lookback
        lookback = 50
        if i >= lookback:
            bbw_slice = bb_width[i-lookback:i+1]
            bbw_valid = bbw_slice[~np.isnan(bbw_slice)]
            if len(bbw_valid) > 0:
                bbw_percentile = (bbw <= np.percentile(bbw_valid, 20)) if not np.isnan(bbw) else False
            else:
                bbw_percentile = False
        else:
            bbw_percentile = False
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: BB squeeze + RSI oversold (<30) + price above 1d EMA50 + volume
            if bbw_percentile and rsi_val < 30 and price > ema_trend and volume_confirm:
                position = 1
                signals[i] = position_size
            # Short: BB squeeze + RSI overbought (>70) + price below 1d EMA50 + volume
            elif bbw_percentile and rsi_val > 70 and price < ema_trend and volume_confirm:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI > 50 or BB width expands (exit squeeze)
            if rsi_val > 50 or (i >= lookback and not bbw_percentile):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI < 50 or BB width expands (exit squeeze)
            if rsi_val < 50 or (i >= lookback and not bbw_percentile):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_BB_Squeeze_RSI_Trend"
timeframe = "4h"
leverage = 1.0