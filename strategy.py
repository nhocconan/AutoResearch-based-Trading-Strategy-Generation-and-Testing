#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla Pivot Breakout with 4h EMA50 trend filter and session filter (08-20 UTC)
# Camarilla pivots provide intraday support/resistance levels based on previous day's range
# Breakout above R3 or below S3 with 4h trend alignment captures strong moves
# Volume confirmation (>1.3x 20-period average) ensures participation
# Session filter (08-20 UTC) reduces noise during low-liquidity hours
# Designed for ~20-40 trades/year on 1h timeframe to minimize fee drag
# Uses 4h trend filter to avoid counter-trend trades in strong markets

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA50 trend filter (HTF = 4h)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla pivots using previous day's OHLC (1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels
    # R4 = Close + ((High-Low)*1.1/2)
    # R3 = Close + ((High-Low)*1.1/4)
    # S3 = Close - ((High-Low)*1.1/4)
    # S4 = Close - ((High-Low)*1.1/2)
    hl_range = prev_high - prev_low
    camarilla_r3 = prev_close + (hl_range * 1.1 / 4)
    camarilla_s3 = prev_close - (hl_range * 1.1 / 4)
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 20-period average volume for confirmation (on 1h data)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade between 08:00 and 20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_ema50_4h = ema_50_4h_aligned[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price drops below EMA50_4h or touches S3 (mean reversion)
            if curr_close < curr_ema50_4h or curr_low <= curr_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price rises above EMA50_4h or touches R3 (mean reversion)
            if curr_close > curr_ema50_4h or curr_high >= curr_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.3x 20-period average
            vol_confirm = curr_volume > 1.3 * curr_vol_ma
            
            # Long entry: price breaks above R3 with 4h uptrend (price > EMA50_4h)
            if vol_confirm and curr_high > curr_r3 and curr_close > curr_ema50_4h:
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below S3 with 4h downtrend (price < EMA50_4h)
            elif vol_confirm and curr_low < curr_s3 and curr_close < curr_ema50_4h:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals