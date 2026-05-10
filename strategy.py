#!/usr/bin/env python3
"""
1h_Momentum_Reversal_Volume_Spike_4hTrend
Hypothesis: In strong 4h trends, 1h momentum reversals with volume spikes provide high-probability entries.
Works in bull/bear: captures trend continuations after pullbacks. Uses 4h trend filter + 1h RSI reversal + volume confirmation.
Target: 20-40 trades/year per symbol (80-160 total) to stay below 200-trade limit and minimize fee drag.
"""

name = "1h_Momentum_Reversal_Volume_Spike_4hTrend"
timeframe = "1h"
leverage = 1.0

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
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema50_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 50:
        ema50_4h[49] = np.mean(close_4h[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_4h)):
            ema50_4h[i] = alpha * close_4h[i] + (1 - alpha) * ema50_4h[i-1]
    
    # 1h RSI(14) for momentum reversal
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    for i in range(14, len(close)):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # 1h volume spike detection (volume > 2x 20-period average)
    vol_sma20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        if i == 20:
            vol_sma20[i] = np.mean(volume[:20])
        else:
            vol_sma20[i] = (vol_sma20[i-1] * 19 + volume[i]) / 20
    volume_spike = volume > (2 * vol_sma20)
    
    # Align 4h trend to 1h timeframe
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Wait for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema50_4h_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_sma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend direction from 4h EMA50
        is_uptrend = close[i] > ema50_4h_aligned[i]
        is_downtrend = close[i] < ema50_4h_aligned[i]
        
        # Momentum reversal conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_falling = rsi[i] < rsi[i-1]  # RSI declining from overbought
        rsi_rising = rsi[i] > rsi[i-1]   # RSI rising from oversold
        
        if position == 0:
            # Long: 4h uptrend + RSI rising from oversold + volume spike
            if is_uptrend and rsi_rising and rsi[i] < 40 and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend + RSI falling from overbought + volume spike
            elif is_downtrend and rsi_falling and rsi[i] > 60 and volume_spike[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: 4h trend turns down OR RSI overbought
            if not is_uptrend or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: 4h trend turns up OR RSI oversold
            if not is_downtrend or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals