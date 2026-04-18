#!/usr/bin/env python3
"""
1d_RSI_Reversal_Volume_TrendFilter_V1
Hypothesis: Daily RSI extremes (oversold/overbought) provide high-probability reversal points when confirmed by volume spikes and aligned with weekly trend. 
Long when RSI(14) < 30, volume > 2x average, and price above weekly EMA(50). 
Short when RSI(14) > 70, volume > 2x average, and price below weekly EMA(50).
Exit when RSI returns to neutral zone (40-60) or trend reverses.
Designed for low frequency (10-25 trades/year) with high win rate in both bull and bear markets by fading extremes only when volume confirms institutional interest.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate RSI(14) on daily data
    def calculate_rsi(close_prices, period=14):
        delta = np.diff(close_prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close_prices, np.nan)
        avg_loss = np.full_like(close_prices, np.nan)
        
        if len(close_prices) >= period:
            # Initial average
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
            
            # Wilder smoothing
            for i in range(period+1, len(close_prices)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1d = calculate_rsi(close_1d, 14)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA(50) for trend filter
    if len(close_1w) >= 50:
        ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    else:
        ema_1w = np.full_like(close_1w, np.nan)
    
    # Align all data to daily timeframe (since we're using 1d timeframe)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: volume > 2x 20-day average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 50, 20) + 1  # Ensure we have enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_aligned[i]) or np.isnan(ema_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Trend filter: price relative to weekly EMA
        above_weekly_ema = close[i] > ema_1w_aligned[i]
        below_weekly_ema = close[i] < ema_1w_aligned[i]
        
        if position == 0:
            # Long: RSI oversold + volume spike + bullish trend bias
            if rsi_aligned[i] < 30 and vol_confirm and above_weekly_ema:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought + volume spike + bearish trend bias
            elif rsi_aligned[i] > 70 and vol_confirm and below_weekly_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral or trend turns bearish
            if rsi_aligned[i] > 40 or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to neutral or trend turns bullish
            if rsi_aligned[i] < 60 or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_RSI_Reversal_Volume_TrendFilter_V1"
timeframe = "1d"
leverage = 1.0