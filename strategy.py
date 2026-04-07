#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h/1d trend filter and volume confirmation
# Uses RSI(14) extremes for mean reversion entries in ranging markets
# Filters by 4h EMA trend to avoid counter-trend trades
# Uses 1d volume spike to confirm institutional interest
# Timeframe: 1h (primary), 4h (trend), 1d (volume)
# Designed for low frequency (target: 15-30 trades/year) to minimize fee impact
# Works in both bull/bear via trend-filtered mean reversion

name = "1h_rsi_mean_reversion_4h_trend_1d_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ema = pd.Series(gain).ewm(alpha=1/14, adjust=False).values
    loss_ema = pd.Series(loss).ewm(alpha=1/14, adjust=False).values
    rs = gain_ema / (loss_ema + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d volume average (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above 1d average
        vol_confirm = volume[i] > vol_ma_1d_aligned[i]
        
        # Trend filter from 4h EMA
        uptrend = close[i] > ema_4h_aligned[i]
        downtrend = close[i] < ema_4h_aligned[i]
        
        # RSI levels for mean reversion
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit when RSI returns to neutral or trend changes
            if rsi[i] >= 50 or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long position
        elif position == -1:  # Short position
            # Exit when RSI returns to neutral or trend changes
            if rsi[i] <= 50 or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short position
        else:  # Flat, look for entry
            # Mean reversion entries with trend filter and volume confirmation
            if rsi_oversold and uptrend and vol_confirm:
                position = 1
                signals[i] = 0.20
            elif rsi_overbought and downtrend and vol_confirm:
                position = -1
                signals[i] = -0.20
    
    return signals