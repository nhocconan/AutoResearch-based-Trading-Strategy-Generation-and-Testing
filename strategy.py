#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band breakout with weekly trend filter and volume confirmation.
# Long when close > upper band AND price > weekly EMA50 AND volume > 1.5x average volume
# Short when close < lower band AND price < weekly EMA50 AND volume > 1.5x average volume
# Exit when close crosses back below/above the middle band (SMA20)
# Uses Bollinger Bands for volatility-based breakout, weekly EMA for trend filter, volume for confirmation.
# Target: 15-25 trades/year per symbol.

name = "1d_Bollinger_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) on daily data
    bb_length = 20
    bb_mult = 2.0
    
    sma = pd.Series(close).rolling(window=bb_length, min_periods=bb_length).mean().values
    std = pd.Series(close).rolling(window=bb_length, min_periods=bb_length).std().values
    upper_band = sma + bb_mult * std
    lower_band = sma - bb_mult * std
    middle_band = sma  # SMA20
    
    # Weekly trend filter (EMA 50)
    df_1w = get_htf_data(prices, '1w')
    weekly_ema50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    
    # Daily average volume for confirmation (20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_length, 50)  # Ensure BB and weekly EMA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sma[i]) or np.isnan(std[i]) or 
            np.isnan(weekly_ema50_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = upper_band[i]
        lower = lower_band[i]
        middle = middle_band[i]
        weekly_ema = weekly_ema50_aligned[i]
        vol_ma_val = vol_ma[i]
        vol = volume[i]
        
        # Volume confirmation
        volume_confirmed = vol > 1.5 * vol_ma_val
        
        if position == 0:
            # Long entry: price breaks above upper band + weekly uptrend + volume confirmation
            if price > upper and price > weekly_ema and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower band + weekly downtrend + volume confirmation
            elif price < lower and price < weekly_ema and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below middle band
            if price < middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above middle band
            if price > middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals