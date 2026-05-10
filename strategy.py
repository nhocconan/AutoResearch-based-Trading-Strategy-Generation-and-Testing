#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_Breakout_1dTrend_Volume_Momentum
Hypothesis: Price breaks above/below daily Camarilla R4/S4 levels with 1d EMA50 trend filter and volume momentum confirmation.
Trades only in direction of daily trend to work in bull/bear markets. Uses momentum oscillator to avoid chop.
Target: 20-30 trades/year (80-120 total) to minimize fee drag.
"""

name = "4h_Camarilla_Pivot_Breakout_1dTrend_Volume_Momentum"
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
    
    # Daily data for Camarilla pivot levels and trend
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels for each daily bar
    range_1d = high_1d - low_1d
    r4_1d = close_1d + 1.5 * range_1d  # Strong resistance
    s4_1d = close_1d - 1.5 * range_1d  # Strong support
    
    # 1d EMA50 for trend filter
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
    
    # Daily volume momentum (volume > 1.5x 20-period average)
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_momentum_1d = np.where(volume_1d > 1.5 * vol_sma20_1d, 1.0, 0.0)
    
    # 4h RSI(14) for momentum filter (avoid chop)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    if len(gain) >= 14:
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        for i in range(14, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_ma = np.full_like(rsi, np.nan)
    if len(rsi) >= 3:
        rsi_ma[2] = np.mean(rsi[:3])
        for i in range(3, len(rsi)):
            rsi_ma[i] = (rsi_ma[i-1] * 2 + rsi[i]) / 3
    
    # Align all indicators to 4h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_momentum_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_momentum_1d)
    rsi_ma_aligned = align_htf_to_ltf(prices, df_1d, rsi_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50 and RSI
    
    for i in range(start_idx, n):
        if np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_momentum_1d_aligned[i]) or np.isnan(rsi_ma_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume momentum: current day has strong volume
        volume_confirm = vol_momentum_1d_aligned[i] > 0.5
        
        # Trend and momentum filters
        is_uptrend = close[i] > ema50_1d_aligned[i]
        is_downtrend = close[i] < ema50_1d_aligned[i]
        rsi_overbought = rsi_ma_aligned[i] > 70
        rsi_oversold = rsi_ma_aligned[i] < 30
        
        # Price relative to Camarilla levels
        price_above_r4 = close[i] > r4_1d_aligned[i]
        price_below_s4 = close[i] < s4_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R4, in uptrend, not overbought, with volume momentum
            if price_above_r4 and is_uptrend and not rsi_overbought and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4, in downtrend, not oversold, with volume momentum
            elif price_below_s4 and is_downtrend and not rsi_oversold and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price falls back below R4 or trend turns down or overbought
            if not price_above_r4 or not is_uptrend or rsi_overbought:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price rises back above S4 or trend turns up or oversold
            if not price_below_s4 or not is_downtrend or rsi_oversold:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals