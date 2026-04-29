#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA50 trend filter and volume confirmation
# Long when Bull Power > 0 AND price > 1d EMA50 AND volume > 1.8x 20-bar avg
# Short when Bear Power < 0 AND price < 1d EMA50 AND volume > 1.8x 20-bar avg
# Exit on opposite zero-cross OR ATR-based stoploss (2.0x ATR)
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 12-30 trades/year on 6h.
# Elder Ray measures bull/bear strength via EMA13. 1d EMA50 filters counter-trend moves.
# Volume spike confirms institutional participation. ATR stoploss manages risk in volatile markets.
# This strategy avoids overtrading by requiring confluence of 3 strong conditions.

name = "6h_ElderRay_1dEMA50_VolumeSpike_ATRStop_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(50) on 1d data
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    # Using 13-period EMA as per standard Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power
    bear_power = low - ema_13   # Bear Power
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 20, 13)  # EMA50(1d), volume MA, EMA13 all need warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        ema_50 = ema_50_1d_aligned[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        atr_val = atr[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Check stoploss: close < entry_price - 2.0 * ATR
            if curr_close < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            # Check exit: Bull Power <= 0 (momentum fading)
            elif bull_val <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Check stoploss: close > entry_price + 2.0 * ATR
            if curr_close > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            # Check exit: Bear Power >= 0 (momentum fading)
            elif bear_val >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when Bull Power > 0 AND price > 1d EMA50 AND volume confirmation
            if bull_val > 0 and curr_close > ema_50 and vol_conf:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short when Bear Power < 0 AND price < 1d EMA50 AND volume confirmation
            elif bear_val < 0 and curr_close < ema_50 and vol_conf:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals