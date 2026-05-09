#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with weekly pivot structure (R2/S2) as trend filter,
# daily RSI(14) for overbought/oversold conditions, and volume confirmation.
# Weekly pivot provides structural support/resistance that works in both bull/bear markets.
# RSI avoids chasing extremes, volume confirms institutional participation.
# Target: 20-60 trades/year (80-240 over 4 years) to minimize fee drag.
name = "6h_WeeklyPivot_R2S2_RSI14_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation and daily data for RSI
    df_weekly = get_htf_data(prices, '1w')
    df_daily = get_htf_data(prices, '1d')
    
    if len(df_weekly) < 50 or len(df_daily) < 50:
        return np.zeros(n)
    
    # Weekly pivot points from previous week's OHLC
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    prev_weekly_high = np.roll(weekly_high, 1)
    prev_weekly_low = np.roll(weekly_low, 1)
    prev_weekly_close = np.roll(weekly_close, 1)
    prev_weekly_high[0] = np.nan
    prev_weekly_low[0] = np.nan
    prev_weekly_close[0] = np.nan
    
    prev_weekly_range = prev_weekly_high - prev_weekly_low
    weekly_pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3
    r2 = weekly_pivot + (prev_weekly_high - prev_weekly_low)
    s2 = weekly_pivot - (prev_weekly_high - prev_weekly_low)
    
    # Daily RSI(14)
    close_daily = pd.Series(df_daily['close'].values)
    delta = close_daily.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume average (24-period for 6h = 4 days)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Align weekly R2/S2 and daily RSI to 6h timeframe
    r2_6h = align_htf_to_ltf(prices, df_weekly, r2)
    s2_6h = align_htf_to_ltf(prices, df_weekly, s2)
    rsi_6h = align_htf_to_ltf(prices, df_daily, rsi_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or np.isnan(rsi_6h[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 24-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        if position == 0:
            # Long: Price above weekly R2 with RSI < 70 (not overbought) and volume spike
            if close[i] > r2_6h[i] and rsi_6h[i] < 70 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Price below weekly S2 with RSI > 30 (not oversold) and volume spike
            elif close[i] < s2_6h[i] and rsi_6h[i] > 30 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below weekly S2 OR RSI > 70 (overbought)
            if close[i] < s2_6h[i] or rsi_6h[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above weekly R2 OR RSI < 30 (oversold)
            if close[i] > r2_6h[i] or rsi_6h[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals