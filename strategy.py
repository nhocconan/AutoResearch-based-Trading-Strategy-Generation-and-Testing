#!/usr/bin/env python3
"""
1d_1w_momentum_reversal_volume_filter_v1
Hypothesis: On daily timeframe, capture momentum reversals using RSI extremes with weekly trend filter and volume confirmation. Weekly trend avoids counter-trend trades, RSI(14)<30 or >70 identifies overextended moves, and volume confirmation ensures institutional participation. Designed for 30-100 total trades over 4 years (~7-25/year) to minimize fee drag while performing in both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_momentum_reversal_volume_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI calculation (14-period)
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Use Wilder's smoothing (alpha = 1/period)
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        
        # Initial values
        if len(gain) >= period:
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
            
            # Wilder's smoothing
            for i in range(period + 1, len(prices)):
                avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
                avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Calculate RSI
    rsi = calculate_rsi(close, 14)
    
    # Weekly trend filter: EMA(21) on weekly data
    df_weekly = get_htf_data(prices, '1w')
    weekly_close = df_weekly['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema)
    
    # Volume filter: 20-day average volume
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(21, 20), n):
        # Skip if data not available
        if (np.isnan(rsi[i]) or np.isnan(weekly_ema_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation: require volume above average
        vol_ok = volume[i] > vol_ma[i]
        
        # Weekly trend: price above/below weekly EMA
        weekly_uptrend = close[i] > weekly_ema_aligned[i]
        weekly_downtrend = close[i] < weekly_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: RSI returns to neutral (50) or weekly trend turns down
            if rsi[i] >= 50 or not weekly_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI returns to neutral (50) or weekly trend turns up
            if rsi[i] <= 50 or not weekly_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Oversold bounce in weekly uptrend: RSI < 30
                if rsi[i] < 30 and weekly_uptrend:
                    position = 1
                    signals[i] = 0.25
                # Overbought reversal in weekly downtrend: RSI > 70
                elif rsi[i] > 70 and weekly_downtrend:
                    position = -1
                    signals[i] = -0.25
    
    return signals