#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_rsi_v1
Hypothesis: On 12-hour timeframe, trade reversals at daily Camarilla pivot levels (H3/L3) with daily RSI confirmation and volume filter. Works in both bull and bear markets by fading extremes at key support/resistance levels. Targets 15-25 trades/year to minimize fee drag while capturing high-probability reversals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_rsi_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot points and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    d_volume = df_1d['volume'].values
    
    # Calculate RSI(14) on daily closes
    def calculate_rsi(close_prices, period=14):
        delta = np.diff(close_prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close_prices)
        avg_loss = np.zeros_like(close_prices)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period + 1, len(close_prices)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(d_close, 14)
    
    # Calculate Camarilla pivot levels for each day
    # H3 = close + (high - low) * 1.1 / 4
    # L3 = close - (high - low) * 1.1 / 4
    cam_h3 = d_close + (d_high - d_low) * 1.1 / 4
    cam_l3 = d_close - (d_high - d_low) * 1.1 / 4
    
    # Align daily data to 12h timeframe
    cam_h3_aligned = align_htf_to_ltf(prices, df_1d, cam_h3)
    cam_l3_aligned = align_htf_to_ltf(prices, df_1d, cam_l3)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume filter: 12h volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = vol_series / vol_ma
    vol_ratio = vol_ratio.fillna(0).values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after volume MA warmup
        # Skip if daily data not available
        if np.isnan(cam_h3_aligned[i]) or np.isnan(cam_l3_aligned[i]) or np.isnan(rsi_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Price proximity to pivot levels (within 0.5% tolerance)
        tolerance = 0.005
        near_h3 = abs(close[i] - cam_h3_aligned[i]) / cam_h3_aligned[i] <= tolerance
        near_l3 = abs(close[i] - cam_l3_aligned[i]) / cam_l3_aligned[i] <= tolerance
        
        # RSI conditions: oversold (<30) for long, overbought (>70) for short
        rsi_oversold = rsi_aligned[i] < 30
        rsi_overbought = rsi_aligned[i] > 70
        
        # Volume confirmation
        vol_confirmed = vol_ratio[i] > 1.5
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit when price reaches H3 or RSI exceeds 50
            if near_h3 or rsi_aligned[i] > 50:
                exit_long = True
            # Exit when volume drops
            elif vol_ratio[i] < 1.0:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit when price reaches L3 or RSI drops below 50
            if near_l3 or rsi_aligned[i] < 50:
                exit_short = True
            # Exit when volume drops
            elif vol_ratio[i] < 1.0:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: near L3 AND RSI oversold AND volume confirmed
            long_entry = near_l3 and rsi_oversold and vol_confirmed
            
            # Short entry: near H3 AND RSI overbought AND volume confirmed
            short_entry = near_h3 and rsi_overbought and vol_confirmed
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals