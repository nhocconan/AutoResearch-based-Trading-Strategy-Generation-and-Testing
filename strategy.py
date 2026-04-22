#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly EMA (34) trend filter + daily RSI (14) mean reversion + volume confirmation.
# Uses weekly EMA to identify long-term trend direction (bull/bear).
# In bull trend: look for RSI < 30 (oversold) with volume spike for long entries.
# In bear trend: look for RSI > 70 (overbought) with volume spike for short entries.
# Designed to work in both bull and bear markets by adapting to weekly trend.
# Targets 10-30 trades/year with disciplined risk control.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for EMA (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA (34)
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Load daily data for RSI and volume
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily RSI (14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rs = np.where(avg_loss == 0, 0, rs)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 20-period average volume for volume spike detection
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume_1d[i]
        vol_ma = vol_ma_20[i]
        ema_val = ema_1w_aligned[i]
        rsi_val = rsi[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Determine trend based on weekly EMA
            bull_trend = price > ema_val
            bear_trend = price < ema_val
            
            if bull_trend:
                # Bull trend: look for oversold conditions to go long
                if rsi_val < 30 and vol_spike:
                    signals[i] = 0.25
                    position = 1
            elif bear_trend:
                # Bear trend: look for overbought conditions to go short
                if rsi_val > 70 and vol_spike:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on overbought RSI or trend change
                if rsi_val > 70 or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on oversold RSI or trend change
                if rsi_val < 30 or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WeeklyEMA34_RSI14_Volume"
timeframe = "1d"
leverage = 1.0