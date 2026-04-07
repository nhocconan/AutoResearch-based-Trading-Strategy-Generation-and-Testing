#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Camarilla Pivot + Daily Trend + Volume Confirmation
# Hypothesis: Camarilla pivot levels (R3/S3 for reversal, R4/S4 for breakout) on 12h timeframe
# combined with daily trend filter and volume confirmation captures institutional activity at key levels.
# Works in bull via R4 breakouts, in bear via S4 breakdowns, and ranges via R3/S3 reversals.
# Target: 12-37 trades/year to minimize fee drag.
name = "12h_camarilla_pivot_daily_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # Using formula: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    # where C = (H+L+C)/3 (typical price)
    prev_close = df_1d['close'].shift(1).values  # Previous day close
    prev_high = df_1d['high'].shift(1).values    # Previous day high
    prev_low = df_1d['low'].shift(1).values      # Previous day low
    
    # Typical price (pivot point)
    pp = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r4 = pp + range_hl * 1.1 / 2.0
    r3 = pp + range_hl * 1.1 / 4.0
    s3 = pp - range_hl * 1.1 / 4.0
    s4 = pp - range_hl * 1.1 / 2.0
    
    # Align Camarilla levels to 12h timeframe
    r4_12h = align_htf_to_ltf(prices, df_1d, r4)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    s4_12h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Get daily trend filter
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=21, adjust=False).mean().values
    daily_ema_12h = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # Volume confirmation: 12h volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(r4_12h[i]) or np.isnan(r3_12h[i]) or 
            np.isnan(s3_12h[i]) or np.isnan(s4_12h[i]) or
            np.isnan(daily_ema_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below S3 (reversal level) or daily trend turns bearish
            if close[i] < s3_12h[i] or close[i] < daily_ema_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above R3 (reversal level) or daily trend turns bullish
            if close[i] > r3_12h[i] or close[i] > daily_ema_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price closes above R4 (breakout) with volume and bullish daily trend
            if close[i] > r4_12h[i] and vol_confirm and close[i] > daily_ema_12h[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below S4 (breakdown) with volume and bearish daily trend
            elif close[i] < s4_12h[i] and vol_confirm and close[i] < daily_ema_12h[i]:
                position = -1
                signals[i] = -0.25
            # Enter long: price closes above R3 (reversal from oversold) with volume
            elif close[i] > r3_12h[i] and vol_confirm and close[i] < daily_ema_12h[i]:
                # Counter-trend long in bearish daily trend
                position = 1
                signals[i] = 0.20
            # Enter short: price closes below S3 (reversal from overbought) with volume
            elif close[i] < s3_12h[i] and vol_confirm and close[i] > daily_ema_12h[i]:
                # Counter-trend short in bullish daily trend
                position = -1
                signals[i] = -0.20
    
    return signals