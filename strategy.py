#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 Breakout with 12h EMA50 trend filter and volume spike
# Long when price breaks above R3, 12h EMA50 up-trend, and volume > 2.0x average
# Short when price breaks below S3, 12h EMA50 down-trend, and volume > 2.0x average
# Exit when price reverts to Camarilla Pivot Point
# Uses discrete position sizing (0.25) to balance capture and risk.
# Tighter volume confirmation (2.0x) reduces trades vs 1.5x, targeting 20-40 trades/year.
# Camarilla R3/S3 are stronger breakout levels than R1/S1, reducing false signals.
# 12h EMA50 provides strong trend filter suitable for 4h timeframe.

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Volume MA and 12h EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels for today (using previous day's OHLC)
        # Need to group by day to get previous day's OHLC
        if i < 1:
            continue
            
        # Get current date from open_time
        curr_date = prices.iloc[i]['open_time'].date()
        prev_date = curr_date  # Initialize
        
        # Find previous trading day's OHLC
        lookback = 0
        found_prev = False
        while lookback < 10 and not found_prev:  # Look back max 10 bars
            check_idx = i - lookback - 1
            if check_idx < 0:
                break
            check_date = prices.iloc[check_idx]['open_time'].date()
            if check_date != curr_date:
                prev_date = check_date
                found_prev = True
                break
            lookback += 1
        
        if not found_prev:
            signals[i] = 0.0
            continue
            
        # Filter prices for previous day
        mask = pd.to_datetime(prices['open_time']).dt.date == prev_date
        if not mask.any():
            signals[i] = 0.0
            continue
            
        prev_day_high = prices.loc[mask, 'high'].max()
        prev_day_low = prices.loc[mask, 'low'].min()
        prev_day_close = prices.loc[mask, 'close'].iloc[-1]
        
        # Calculate Camarilla levels
        range_val = prev_day_high - prev_day_low
        if range_val <= 0:
            signals[i] = 0.0
            continue
            
        camarilla_pivot = (prev_day_high + prev_day_low + prev_day_close) / 3
        r3 = camarilla_pivot + (range_val * 1.1 / 4)
        s3 = camarilla_pivot - (range_val * 1.1 / 4)
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_ema50_12h = ema_50_12h_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price below Camarilla Pivot
            if curr_close < camarilla_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above Camarilla Pivot
            if curr_close > camarilla_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 20-period average (tighter than 1.5x)
            vol_confirmed = curr_volume > 2.0 * curr_vol_ma
            
            # Long when price breaks above R3, 12h EMA50 up-trend, volume confirmed
            if curr_close > r3 and curr_close > curr_ema50_12h and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S3, 12h EMA50 down-trend, volume confirmed
            elif curr_close < s3 and curr_close < curr_ema50_12h and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals