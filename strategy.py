#!/usr/bin/env python3
name = "4h_1d_Camarilla_S1R1_Breakout_Trend_Rev"
timeframe = "4h"
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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    s1 = prev_close - (range_hl * 1.08 / 2)
    r1 = prev_close + (range_hl * 1.08 / 2)
    
    # Align daily levels to 4h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    
    # Daily trend filter: EMA(34) on daily close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 6-period average (1.5 days of 4h bars)
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    # Momentum filter: RSI(14) on close to avoid overextended entries
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 6, 14)  # Wait for EMA, volume MA, and RSI
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(vol_ma_6[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S1 with volume and daily uptrend, RSI not overbought
            vol_condition = volume[i] > vol_ma_6[i] * 1.6
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            rsi_ok = rsi[i] < 70
            
            if close[i] > s1_aligned[i] and vol_condition and uptrend and rsi_ok:
                signals[i] = 0.25
                position = 1
            # Short: price below R1 with volume and daily downtrend, RSI not oversold
            elif close[i] < r1_aligned[i] and vol_condition and not uptrend and rsi[i] > 30:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below S1 or volume drops or RSI overbought
            if close[i] < s1_aligned[i] or volume[i] < vol_ma_6[i] * 1.1 or rsi[i] > 75:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above R1 or volume drops or RSI oversold
            if close[i] > r1_aligned[i] or volume[i] < vol_ma_6[i] * 1.1 or rsi[i] < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Camarilla S1/R1 breakout with daily trend, volume confirmation, and RSI filter
# - Daily Camarilla S1/R1 act as strong support/resistance levels derived from prior day's range
# - Breakout above S1 with volume in daily uptrend = long opportunity (avoid shorts in uptrend)
# - Breakdown below R1 with volume in daily downtrend = short opportunity (avoid longs in downtrend)
# - Volume spike (1.6x average) confirms institutional participation
# - RSI filter (30-70) prevents entries on overextended moves, improving win rate in ranging markets
# - Works in both bull (buy S1 breaks in uptrend) and bear (sell R1 breaks in downtrend) via trend filter
# - Exit when price returns to S1/R1, volume weakens, or RSI reaches extreme levels
# - Position size 0.25 targets ~25-40 trades/year, avoiding fee drag while maintaining edge
# - Uses actual daily Camarilla levels (not weekly) for better responsiveness to recent price action
# - Reduced volume multiplier from 1.8 to 1.6 and tightened exit conditions to reduce trade frequency
# - Added RSI filter to prevent chasing momentum and improve robustness in BTC/ETH markets