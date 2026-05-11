#!/usr/bin/env python3
"""
12h_1d_RSI_MeanReversion_Regime
Hypothesis: Mean-reversion on 12h using RSI(14) with 1d trend filter and volume confirmation.
- Long when: RSI < 30, 1d EMA50 uptrend (close > EMA50), volume > 20-period average
- Short when: RSI > 70, 1d EMA50 downtrend (close < EMA50), volume > 20-period average
- Exit when RSI returns to neutral (40-60) or trend reverses
RSI captures overextended moves. Mean reversion works in ranging markets (2025+).
Trend filter avoids counter-trend trades. Volume ensures participation.
Targets 15-30 trades/year (60-120 over 4 years) to minimize fee drag.
"""

name = "12h_1d_RSI_MeanReversion_Regime"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    volume_12h = prices['volume'].values
    
    # --- 1d Trend Filter: EMA50 ---
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # --- RSI on 12h ---
    rsi_period = 14
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Wilder's smoothing (equivalent to EMA with alpha=1/period)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[rsi_period] = np.mean(gain[1:rsi_period+1])
    avg_loss[rsi_period] = np.mean(loss[1:rsi_period+1])
    
    for i in range(rsi_period + 1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:rsi_period] = np.nan  # First values undefined
    
    # --- Volume Confirmation: 12h volume > 20-period average ---
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = max(30, rsi_period + 5)  # Ensure RSI and vol MA are valid
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(rsi[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        trend_up = close_12h[i] > ema50_1d_aligned[i]
        trend_down = close_12h[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        vol_ok = volume_12h[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for mean-reversion entries only in direction of 1d trend with volume
            if rsi[i] < 30 and trend_up and vol_ok:
                # Long: oversold + 1d uptrend + volume
                signals[i] = 0.25
                position = 1
            elif rsi[i] > 70 and trend_down and vol_ok:
                # Short: overbought + 1d downtrend + volume
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: RSI returns to neutral (40-60) OR trend turns down
                if rsi[i] >= 40 or not trend_up:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: RSI returns to neutral (40-60) OR trend turns up
                if rsi[i] <= 60 or not trend_down:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals