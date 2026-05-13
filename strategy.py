#!/usr/bin/env python3
"""
4h_PowerTrend_Zone_Scalper
Hypothesis: Combines 4h EMA crossovers (8/21) with volume-weighted RSI to catch momentum in both bull and bear markets.
Uses price position relative to EMA bands for trend strength and RSI for momentum confirmation.
Designed for low trade frequency (15-30/year) with clear entry/exit rules to minimize fee drag.
Works in bull markets via trend continuation and in bear markets via mean-reversion within trends.
"""

name = "4h_PowerTrend_Zone_Scalper"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate EMAs for trend (8, 21)
    ema_8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate RSI with volume weighting (14-period)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Volume-weighted average gain/loss
    vol_weight = pd.Series(volume).ewm(span=14, adjust=False, min_periods=14).mean().values
    vol_gain = (gain * vol_weight).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    vol_loss = (loss * vol_weight).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = vol_gain / vol_loss.replace(0, np.nan)
    rsi = (100 - (100 / (1 + rs))).fillna(50).values
    
    # Get 1-day trend filter (EMA 50) for higher timeframe bias
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(21, n):
        if position == 0:
            # LONG: EMA 8 > EMA 21 (uptrend) + RSI > 55 (bullish momentum) + volume confirmation
            if ema_8[i] > ema_21[i] and rsi[i] > 55 and volume_confirm[i]:
                # Additional filter: price above 1-day EMA50 (strong uptrend filter)
                if close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # SHORT: EMA 8 < EMA 21 (downtrend) + RSI < 45 (bearish momentum) + volume confirmation
            elif ema_8[i] < ema_21[i] and rsi[i] < 45 and volume_confirm[i]:
                # Additional filter: price below 1-day EMA50 (strong downtrend filter)
                if close[i] < ema_50_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: EMA cross down OR RSI < 40 (momentum fade)
            if ema_8[i] < ema_21[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: EMA cross up OR RSI > 60 (momentum fade)
            if ema_8[i] > ema_21[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals