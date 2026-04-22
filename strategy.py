#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) with 4h EMA(50) trend filter and volume confirmation.
# RSI identifies overbought (>70) and oversold (<30) conditions.
# 4h EMA provides trend direction: only take longs when price > 4h EMA, shorts when price < 4h EMA.
# Volume confirmation requires current volume > 1.5x 20-period average to avoid false signals.
# Session filter (08-20 UTC) reduces noise trades outside active hours.
# Designed to work in both bull and bear markets by aligning with trend via 4h EMA filter.
# Targets 15-37 trades/year with strict entry conditions to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data for EMA trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 50-period EMA on 4h data
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 14-period RSI
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_4h_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_ma_20[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        rsi_val = rsi[i]
        ema_val = ema_4h_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: oversold + uptrend + volume spike
            if rsi_val < 30 and price > ema_val and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short conditions: overbought + downtrend + volume spike
            elif rsi_val > 70 and price < ema_val and vol_spike:
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when RSI returns to overbought or trend breaks
                if rsi_val > 70 or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when RSI returns to oversold or trend breaks
                if rsi_val < 30 or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_RSI_4hEMA_Trend_Volume_Session"
timeframe = "1h"
leverage = 1.0