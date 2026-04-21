#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) Pullback + 4h EMA(50) Trend + 1d Volume Spike
# Long when: 1h RSI < 30 (oversold), price > 4h EMA50 (uptrend), 1d volume > 1.5x 20-day average
# Short when: 1h RSI > 70 (overbought), price < 4h EMA50 (downtrend), 1d volume > 1.5x 20-day average
# Exit when RSI crosses 50 (mean reversion to midpoint)
# Trend filter (4h EMA50) ensures we trade with higher timeframe trend
# Volume spike confirms institutional participation
# RSI pullback provides high-probability entry points within trend
# Target: 15-30 trades/year by requiring multiple confluence conditions

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA(50) for trend
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 1h RSI(14)
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    rsi_period = 14
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    
    avg_gain[rsi_period] = np.mean(gain[1:rsi_period+1])
    avg_loss[rsi_period] = np.mean(loss[1:rsi_period+1])
    
    for i in range(rsi_period+1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:rsi_period+1] = 50  # Neutral before enough data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price
        price = prices['close'].iloc[i]
        
        # Volume confirmation: current 1d volume > 1.5x 20-day average
        # Get today's volume from 1d data
        days_elapsed = i // 24  # Approximate days from start
        if days_elapsed < len(df_1d):
            todays_volume = df_1d['volume'].iloc[days_elapsed]
        else:
            todays_volume = df_1d['volume'].iloc[-1] if len(df_1d) > 0 else 0
        
        vol_ma = vol_ma_1d_aligned[i]
        volume_confirm = todays_volume > 1.5 * vol_ma
        
        if position == 0:
            # Long setup: RSI oversold, price above 4h EMA50, volume spike
            if rsi[i] < 30 and price > ema_50_4h_aligned[i] and volume_confirm:
                signals[i] = 0.20
                position = 1
            # Short setup: RSI overbought, price below 4h EMA50, volume spike
            elif rsi[i] > 70 and price < ema_50_4h_aligned[i] and volume_confirm:
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit when RSI crosses 50 (mean reversion)
            exit_signal = False
            
            if position == 1:  # long position
                if rsi[i] > 50:
                    exit_signal = True
            
            elif position == -1:  # short position
                if rsi[i] < 50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_RSI14_Pullback_4hEMA50_Trend_1dVolumeSpike"
timeframe = "1h"
leverage = 1.0