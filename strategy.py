#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla R3/S3 breakout + 1w EMA34 trend + volume spike
- Long: Close breaks above Camarilla R3 + price > 1w EMA34 (uptrend) + volume > 2.0x 20-period average
- Short: Close breaks below Camarilla S3 + price < 1w EMA34 (downtrend) + volume > 2.0x 20-period average
- Exit: Close retouches Camarilla H3/L3 OR trend reversal
- Uses discrete position sizing (0.25) to minimize fee churn
- Target: 15-30 trades/year (60-120 over 4 years) to avoid fee drag
- Camarilla levels provide intraday structure; breakouts with volume and HTF trend filter work in both bull and bear markets
- Using 1w EMA34 as HTF trend filter for better alignment with 1d timeframe
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate Camarilla levels from 1d data (using previous day's OHLC)
    # Camarilla formula: 
    # H4 = close + 1.5*(high-low), H3 = close + 1.0*(high-low), L3 = close - 1.0*(high-low), L4 = close - 1.5*(high-low)
    # But we use daily pivot: P = (high + low + close)/3
    # Then R3 = P + 1.0*(high-low), S3 = P - 1.0*(high-low)
    # We'll use previous day's data to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    camarilla_r3 = pivot + 1.0 * (prev_high - prev_low)
    camarilla_s3 = pivot - 1.0 * (prev_high - prev_low)
    camarilla_h3 = pivot + 0.5 * (prev_high - prev_low)  # Exit level
    camarilla_l3 = pivot - 0.5 * (prev_high - prev_low)  # Exit level
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 1)  # EMA34 needs 34, but we use prev data so +1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_aligned[i]) or 
            np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or 
            np.isnan(camarilla_h3[i]) or 
            np.isnan(camarilla_l3[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 1w EMA34
        uptrend = close[i] > ema34_aligned[i]
        downtrend = close[i] < ema34_aligned[i]
        
        # Camarilla breakout signals with trend filter and volume confirmation
        # Long: Close breaks above R3 + uptrend + volume spike
        # Short: Close breaks below S3 + downtrend + volume spike
        long_signal = (close[i] > camarilla_r3[i] and 
                      uptrend and
                      volume[i] > 2.0 * vol_ma[i])
        
        short_signal = (close[i] < camarilla_s3[i] and 
                       downtrend and
                       volume[i] > 2.0 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Close retouches H3/L3 OR trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: Close retouches H3 or trend turns down
                if (close[i] <= camarilla_h3[i] or 
                    not uptrend):
                    exit_signal = True
            elif position == -1:
                # Exit short: Close retouches L3 or trend turns up
                if (close[i] >= camarilla_l3[i] or 
                    not downtrend):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Camarilla_R3S3_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0