#!/usr/bin/env python3
"""
6h_RangeBound_MeanReversion
Hypothesis: In low-volatility regimes (choppy markets), price mean-reverts from daily Bollinger Band extremes (2 std dev). 
Enter long at lower band when RSI < 30 and volume > 1.5x 20-day average; enter short at upper band when RSI > 70 and volume > 1.5x average. 
Use 12h EMA50 trend filter to avoid trading against strong trends. Works in both bull/bear by capturing reversion within ranges.
Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
"""

name = "6h_RangeBound_MeanReversion"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Bollinger Bands and RSI
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma20_1d = np.full(len(close_1d), np.nan)
    std20_1d = np.full(len(close_1d), np.nan)
    
    if len(close_1d) >= bb_period:
        for i in range(bb_period - 1, len(close_1d)):
            sma20_1d[i] = np.mean(close_1d[i - bb_period + 1:i + 1])
            std20_1d[i] = np.std(close_1d[i - bb_period + 1:i + 1])
    
    upper_bb = sma20_1d + bb_std * std20_1d
    lower_bb = sma20_1d - bb_std * std20_1d
    
    # RSI(14)
    rsi_period = 14
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close_1d), np.nan)
    avg_loss = np.full(len(close_1d), np.nan)
    
    if len(close_1d) >= rsi_period:
        avg_gain[rsi_period - 1] = np.mean(gain[:rsi_period])
        avg_loss[rsi_period - 1] = np.mean(loss[:rsi_period])
        for i in range(rsi_period, len(close_1d)):
            avg_gain[i] = (avg_gain[i - 1] * (rsi_period - 1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i - 1] * (rsi_period - 1) + loss[i]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        ema50_12h[49] = np.mean(close_12h[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_12h)):
            ema50_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema50_12h[i - 1]
    
    # Daily volume SMA20
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i - 1] * 19 + volume_1d[i]) / 20
    
    # Align all indicators to 6h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, bb_period, rsi_period)  # Wait for indicators
    
    for i in range(start_idx, n):
        if np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average daily volume (scaled)
        # 1d = 4 x 6h bars, so scale daily volume to 6h equivalent
        vol_1d_scaled = vol_sma20_1d_aligned[i] / 4.0  # Average 6h-equivalent volume from 1d data
        volume_confirm = volume[i] > 1.5 * vol_1d_scaled
        
        # Trend and price relative to Bollinger Bands
        is_uptrend = close[i] > ema50_12h_aligned[i]
        is_downtrend = close[i] < ema50_12h_aligned[i]
        price_at_lower = close[i] <= lower_bb_aligned[i]
        price_at_upper = close[i] >= upper_bb_aligned[i]
        rsi_oversold = rsi_1d_aligned[i] < 30
        rsi_overbought = rsi_1d_aligned[i] > 70
        
        if position == 0:
            # Long: price at lower band, oversold, volume, not in strong uptrend
            if price_at_lower and rsi_oversold and volume_confirm and not is_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price at upper band, overbought, volume, not in strong downtrend
            elif price_at_upper and rsi_overbought and volume_confirm and not is_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price moves back to middle band or RSI normalizes
            if close[i] >= sma20_1d_aligned[i] or rsi_1d_aligned[i] >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price moves back to middle band or RSI normalizes
            if close[i] <= sma20_1d_aligned[i] or rsi_1d_aligned[i] <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals