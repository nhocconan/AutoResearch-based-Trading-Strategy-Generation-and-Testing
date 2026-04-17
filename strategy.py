#!/usr/bin/env python3
"""
6h_AbnormalVolume_Momentum_Reversal_v1
Detect momentum reversals after abnormal volume spikes on 6h timeframe.
Uses volume spike (>2.5x 20-period average) combined with RSI divergence
and price rejection at key levels. Works in both bull/bear markets by
fading exhaustion moves after high-volume spikes.
Target: 50-150 total trades over 4 years (12-37/year).
"""

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
    
    # === Volume spike detection (20-period average) ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 20:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1]) if i > 0 else volume[0]
    
    vol_spike = volume > vol_ma_20 * 2.5  # abnormal volume: 2.5x average
    
    # === RSI (14-period) for momentum/divergence ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    for i in range(len(close)):
        if i >= 14:
            if i == 14:
                avg_gain[i] = np.mean(gain[1:15])
                avg_loss[i] = np.mean(loss[1:15])
            else:
                avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
                avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        else:
            avg_gain[i] = np.nan
            avg_loss[i] = np.nan
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # === Price rejection detection ===
    # Bullish rejection: long wick down, close near high
    bullish_rejection = (close - low) > (high - close) * 1.5 and (close - low) > (high - low) * 0.6
    # Bearish rejection: long wick up, close near low
    bearish_rejection = (high - close) > (close - low) * 1.5 and (high - close) > (high - low) * 0.6
    
    # Vectorize the rejection conditions
    bullish_rejection = np.where(
        (close - low) > (high - close) * 1.5,
        np.where((close - low) > (high - low) * 0.6, True, False),
        False
    )
    bearish_rejection = np.where(
        (high - close) > (close - low) * 1.5,
        np.where((high - close) > (high - low) * 0.6, True, False),
        False
    )
    
    # === 12h trend filter (EMA34) for context ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_34_12h = np.full_like(close_12h, np.nan)
    for i in range(len(close_12h)):
        if i >= 34:
            if i == 34:
                ema_34_12h[i] = np.mean(close_12h[1:35])
            else:
                ema_34_12h[i] = (close_12h[i] * 2 + ema_34_12h[i-1] * 33) / 34
        else:
            ema_34_12h[i] = np.nan
    
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: volume spike + bullish rejection + RSI not overbought + price above 12h EMA (uptrend filter)
            if (vol_spike[i] and 
                bullish_rejection[i] and 
                rsi[i] < 70 and 
                close[i] > ema_34_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: volume spike + bearish rejection + RSI not oversold + price below 12h EMA (downtrend filter)
            elif (vol_spike[i] and 
                  bearish_rejection[i] and 
                  rsi[i] > 30 and 
                  close[i] < ema_34_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: RSI overbought OR price closes below 12h EMA
            if (rsi[i] > 70 or 
                close[i] < ema_34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI oversold OR price closes above 12h EMA
            if (rsi[i] < 30 or 
                close[i] > ema_34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_AbnormalVolume_Momentum_Reversal_v1"
timeframe = "6h"
leverage = 1.0