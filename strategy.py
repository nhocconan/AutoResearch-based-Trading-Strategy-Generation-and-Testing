#!/usr/bin/env python3
"""
Hypothesis: 4-hour volume-weighted RSI with 1-day trend filter and volatility regime.
Long when 4h VW-RSI < 30, 1-day EMA34 rising, and volatility low (ATR ratio < 0.8).
Short when 4h VW-RSI > 70, 1-day EMA34 falling, and volatility low.
Exit when VW-RSI returns to 50.
Uses volume-weighted RSI for institutional bias, daily EMA for trend, and volatility filter to avoid whipsaws.
Designed for low trade frequency with clear mean-reversion signals in ranging markets.
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
    
    # Load 1-day data for EMA trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate VW-RSI (Volume Weighted RSI) on 4h data
    # Typical price = (H + L + C) / 3
    typical_price = (high + low + close) / 3.0
    # Price change
    delta = np.diff(typical_price, prepend=typical_price[0])
    # Volume-weighted gains and losses
    gains = np.where(delta > 0, delta * volume, 0.0)
    losses = np.where(delta < 0, -delta * volume, 0.0)
    # Smoothed average gains/losses (Wilder's smoothing)
    avg_gain = np.zeros_like(gains)
    avg_loss = np.zeros_like(losses)
    # Initialize with first 14 period average
    if len(gains) >= 14:
        avg_gain[13] = np.mean(gains[1:15])  # Skip first (no change)
        avg_loss[13] = np.mean(losses[1:15])
        # Wilder smoothing: avg = (prev_avg * 13 + current) / 14
        for i in range(14, len(gains)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gains[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + losses[i]) / 14
    # Calculate RSI
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    vw_rsi = 100 - (100 / (1 + rs))
    # Handle division by zero (when avg_loss = 0)
    vw_rsi = np.where(avg_loss == 0, 100, vw_rsi)
    # And when avg_gain = 0
    vw_rsi = np.where(avg_gain == 0, 0, vw_rsi)
    
    # 1-day EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volatility filter: ATR ratio (current ATR / 20-period ATR average) < 0.8
    # Calculate ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0  # First period has no previous close
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.zeros_like(tr)
    if len(tr) >= 14:
        atr[13] = np.mean(tr[1:15])
        for i in range(14, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    atr_ma_20 = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    atr_ratio = np.where(atr_ma_20 > 0, atr / atr_ma_20, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):  # Start after enough data for all indicators
        # Skip if data not ready
        if (np.isnan(vw_rsi[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(atr_ratio[i]) or np.isnan(atr_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_filter = atr_ratio[i] < 0.8  # Low volatility regime
        
        if position == 0:
            # Long: VW-RSI oversold (<30), 1-day EMA34 rising, low volatility
            if (vw_rsi[i] < 30 and 
                ema34_1d_aligned[i] > ema34_1d_aligned[i-1] and vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: VW-RSI overbought (>70), 1-day EMA34 falling, low volatility
            elif (vw_rsi[i] > 70 and 
                  ema34_1d_aligned[i] < ema34_1d_aligned[i-1] and vol_filter):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: VW-RSI returns to neutral (50)
            exit_signal = False
            
            if position == 1:
                # Exit long: VW-RSI crosses above 50
                if vw_rsi[i] >= 50 and vw_rsi[i-1] < 50:
                    exit_signal = True
            else:  # position == -1
                # Exit short: VW-RSI crosses below 50
                if vw_rsi[i] <= 50 and vw_rsi[i-1] > 50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_VW_RSI_1dEMA34_VolFilter_MeanRev"
timeframe = "4h"
leverage = 1.0