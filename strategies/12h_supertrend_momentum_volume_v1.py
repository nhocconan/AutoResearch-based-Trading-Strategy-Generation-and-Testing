#!/usr/bin/env python3
# 12h_supertrend_momentum_volume_v1
# Hypothesis: 12h Supertrend for trend direction, momentum (RSI) for entry timing, and volume confirmation.
# Works in bull markets by following uptrend and buying pullbacks, and in bear markets by following downtrend and selling rallies.
# Volume filter ensures trades occur with institutional participation, reducing false signals.
# Target: 20-40 trades/year via Supertrend trend filter + RSI momentum + volume confirmation.

name = "12h_supertrend_momentum_volume_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Supertrend (ATR=10, multiplier=3.0)
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # First bar
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR using Wilder's smoothing
    atr = np.zeros_like(tr)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Supertrend calculation
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros_like(close)
    direction = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, len(close)):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower_band[i]
            direction[i] = 1
        else:
            supertrend[i] = upper_band[i]
            direction[i] = -1
    
    # RSI (14-period) for momentum
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[rsi_period-1] = np.mean(gain[:rsi_period])
    avg_loss[rsi_period-1] = np.mean(loss[:rsi_period])
    
    for i in range(rsi_period, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: 20-period average volume
    vol_ma = np.zeros_like(volume)
    vol_ma[19:] = np.convolve(volume, np.ones(20)/20, mode='valid')
    vol_ma[:19] = vol_ma[19]  # Fill beginning with first valid value
    
    # Get daily data for trend filter (higher timeframe)
    df_daily = get_htf_data(prices, '1d')
    close_daily = df_daily['close'].values
    
    # Daily EMA (50-period) for higher timeframe trend
    ema_period = 50
    ema_daily = np.zeros_like(close_daily)
    ema_daily[ema_period-1] = np.mean(close_daily[:ema_period])
    for i in range(ema_period, len(close_daily)):
        ema_daily[i] = (close_daily[i] * 2 + ema_daily[i-1] * (ema_period - 1)) / (ema_period + 1)
    
    # Align daily EMA to 12h timeframe
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = max(atr_period, rsi_period, ema_period) + 5
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(supertrend[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_daily_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        # Higher timeframe trend filter: price above/below daily EMA
        uptrend_htf = close[i] > ema_daily_aligned[i]
        downtrend_htf = close[i] < ema_daily_aligned[i]
        
        if position == 1:  # Long position
            # Exit if trend reverses or momentum fails
            if direction[i] == -1 or rsi[i] < 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if trend reverses or momentum fails
            if direction[i] == 1 or rsi[i] > 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: uptrend on Supertrend, RSI momentum, volume, and HTF uptrend
            if (direction[i] == 1 and 
                rsi[i] > 50 and rsi[i] < 70 and  # Not overbought, but bullish momentum
                volume_filter and 
                uptrend_htf):
                position = 1
                signals[i] = 0.25
            # Short entry: downtrend on Supertrend, RSI momentum, volume, and HTF downtrend
            elif (direction[i] == -1 and 
                  rsi[i] < 50 and rsi[i] > 30 and  # Not oversold, but bearish momentum
                  volume_filter and 
                  downtrend_htf):
                position = -1
                signals[i] = -0.25
    
    return signals