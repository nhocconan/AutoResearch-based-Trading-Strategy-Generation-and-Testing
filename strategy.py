#!/usr/bin/env python3
"""
4h_Triple_Screen_Trading_System
Based on Elder's Triple Screen system:
- Long-term trend (weekly EMA13) filters direction
- Intermediate trend (daily MACD histogram) confirms momentum
- Short-term entry (4h RSI<30 for long, RSI>70 for short) with overbought/oversold
- Exit on opposite RSI extreme or trend reversal
- Designed for low trade frequency (<50/year) with strong edge in both bull/bear markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Triple_Screen_Trading_System"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === WEEKLY TREND FILTER (EMA13) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA13 for trend direction
    weekly_close = df_1w['close'].values
    alpha = 2 / (13 + 1)
    ema_w13 = np.full_like(weekly_close, np.nan, dtype=float)
    ema_w13[0] = weekly_close[0]
    for i in range(1, len(weekly_close)):
        ema_w13[i] = alpha * weekly_close[i] + (1 - alpha) * ema_w13[i-1]
    
    # Weekly trend: bullish if price > EMA13, bearish if price < EMA13
    weekly_bullish = weekly_close > ema_w13
    weekly_bearish = weekly_close < ema_w13
    
    # Forward fill and align to 4h
    weekly_bullish_ffilled = pd.Series(weekly_bullish).ffill().values
    weekly_bearish_ffilled = pd.Series(weekly_bearish).ffill().values
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish_ffilled)
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish_ffilled)
    
    # === DAILY MOMENTUM CONFIRMATION (MACD HISTOGRAM) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Daily MACD: EMA12 - EMA26
    daily_close = df_1d['close'].values
    
    # EMA12
    alpha12 = 2 / (12 + 1)
    ema12 = np.full_like(daily_close, np.nan, dtype=float)
    ema12[0] = daily_close[0]
    for i in range(1, len(daily_close)):
        ema12[i] = alpha12 * daily_close[i] + (1 - alpha12) * ema12[i-1]
    
    # EMA26
    alpha26 = 2 / (26 + 1)
    ema26 = np.full_like(daily_close, np.nan, dtype=float)
    ema26[0] = daily_close[0]
    for i in range(1, len(daily_close)):
        ema26[i] = alpha26 * daily_close[i] + (1 - alpha26) * ema26[i-1]
    
    macd_line = ema12 - ema26
    
    # Signal line: EMA9 of MACD
    alpha9 = 2 / (9 + 1)
    signal_line = np.full_like(macd_line, np.nan, dtype=float)
    valid = ~np.isnan(macd_line)
    if np.any(valid):
        first_valid = np.where(valid)[0][0]
        signal_line[first_valid] = macd_line[first_valid]
        for i in range(first_valid + 1, len(macd_line)):
            if not np.isnan(macd_line[i]):
                signal_line[i] = alpha9 * macd_line[i] + (1 - alpha9) * signal_line[i-1]
            else:
                signal_line[i] = signal_line[i-1]
    
    # MACD histogram: positive = bullish momentum, negative = bearish momentum
    macd_hist = macd_line - signal_line
    daily_bullish_momentum = macd_hist > 0
    daily_bearish_momentum = macd_hist < 0
    
    # Forward fill and align to 4h
    daily_bullish_mom_ffilled = pd.Series(daily_bullish_momentum).ffill().values
    daily_bearish_mom_ffilled = pd.Series(daily_bearish_momentum).ffill().values
    daily_bullish_mom_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish_mom_ffilled)
    daily_bearish_mom_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish_mom_ffilled)
    
    # === SHORT-TERM ENTRY (4H RSI) ===
    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Smoothed gains/losses (Wilder's smoothing)
    alpha_rsi = 1 / 14
    avg_gain = np.full_like(gain, np.nan, dtype=float)
    avg_loss = np.full_like(loss, np.nan, dtype=float)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    for i in range(1, len(gain)):
        avg_gain[i] = alpha_rsi * gain[i] + (1 - alpha_rsi) * avg_gain[i-1]
        avg_loss[i] = alpha_rsi * loss[i] + (1 - alpha_rsi) * avg_loss[i-1]
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi[avg_loss == 0] = 100  # All gains, no losses
    
    # RSI thresholds
    rsi_oversold = rsi < 30
    rsi_overbought = rsi > 70
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]) or 
            np.isnan(daily_bullish_mom_aligned[i]) or np.isnan(daily_bearish_mom_aligned[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI overbought OR weekly trend turns bearish OR daily momentum turns bearish
            if rsi_overbought[i] or weekly_bearish_aligned[i] or not daily_bullish_mom_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: RSI oversold OR weekly trend turns bullish OR daily momentum turns bullish
            if rsi_oversold[i] or weekly_bullish_aligned[i] or not daily_bearish_mom_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: Weekly bullish AND daily bullish momentum AND RSI oversold
            if weekly_bullish_aligned[i] and daily_bullish_mom_aligned[i] and rsi_oversold[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: Weekly bearish AND daily bearish momentum AND RSI overbought
            elif weekly_bearish_aligned[i] and daily_bearish_mom_aligned[i] and rsi_overbought[i]:
                position = -1
                signals[i] = -0.25
    
    return signals