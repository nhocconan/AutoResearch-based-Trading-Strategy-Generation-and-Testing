#!/usr/bin/env python3
name = "12h_1d_Trend_Momentum_1wPivot"
timeframe = "12h"
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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly pivot points (using daily high/low/close)
    weekly_high = df_1d['high'].rolling(window=5, min_periods=5).max().values
    weekly_low = df_1d['low'].rolling(window=5, min_periods=5).min().values
    weekly_close = df_1d['close'].rolling(window=5, min_periods=5).mean().values
    
    # Pivot levels
    pp = (weekly_high + weekly_low + weekly_close) / 3
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    
    # Align weekly pivot levels to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Daily trend: EMA(21) on close
    ema_21_1d = df_1d['close'].ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # Momentum: RSI(14) on 12h closes
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume filter: 24-period average (2 days of 12h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 24, 5)  # Wait for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_21_1d_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma_24[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S1 with bullish momentum and daily uptrend
            vol_condition = volume[i] > vol_ma_24[i] * 1.5
            bullish_momentum = rsi[i] > 50 and rsi[i] < 70  # Avoid overbought
            daily_uptrend = ema_21_1d_aligned[i] > ema_21_1d_aligned[i-1]
            
            if close[i] > s1_aligned[i] and vol_condition and bullish_momentum and daily_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price below R1 with bearish momentum and daily downtrend
            elif close[i] < r1_aligned[i] and vol_condition and (rsi[i] < 50 and rsi[i] > 30) and not daily_uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below S1 or momentum fades
            if close[i] < s1_aligned[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above R1 or momentum fades
            if close[i] > r1_aligned[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h strategy using weekly pivot points as dynamic support/resistance,
# combined with daily trend filter and RSI momentum confirmation.
# - Long when price breaks above S1 (weekly support) with bullish RSI (50-70) and daily uptrend
# - Short when price breaks below R1 (weekly resistance) with bearish RSI (30-50) and daily downtrend
# - Volume confirmation (1.5x average) ensures institutional participation
# - Exit when price returns to pivot level or momentum weakens (RSI < 40 for longs, > 60 for shorts)
# - Works in bull markets (buy S1 breaks in uptrends) and bear markets (sell R1 breaks in downtrends)
# - Position size 0.25 targets 15-35 trades/year, avoiding excessive fee drag
# - Weekly pivot provides structural support/resistance that adapts to volatility regimes
# - RSI momentum filter prevents chasing overextended moves
# - Daily trend filter ensures alignment with higher timeframe bias