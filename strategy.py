#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h EMA50 trend filter and volume spike confirmation
# Uses 4h EMA50 for trend direction (price > EMA50 = bullish, < EMA50 = bearish)
# Entry on break of Camarilla R3/S3 levels with volume > 2.0x 20-period average
# Designed for ~15-37 trades/year on 1h timeframe to minimize fee drag
# Session filter (08-20 UTC) to avoid low-liquidity periods
# Works in both bull/bear via 4h EMA50 trend filter - only trades in direction of higher timeframe momentum

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Get 4h data for EMA50 trend filter (HTF = 4h)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate ATR (14-period) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla pivot points from previous day
    # Need daily high/low/close for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels: based on previous day's range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3, R4, S3, S4 levels
    # R4 = close + 1.5*(high-low)
    # R3 = close + 1.1*(high-low)
    # S3 = close - 1.1*(high-low)
    # S4 = close - 1.5*(high-low)
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    start_idx = 20  # volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_ema50_4h = ema_50_4h_aligned[i]
        curr_atr = atr[i]
        curr_vol_ma = vol_ma_20[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = curr_volume > 2.0 * curr_vol_ma
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: stoploss hit or price breaks below S3 (failure of bullish breakout)
            if curr_close < entry_price - 1.5 * curr_atr or curr_close < curr_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: stoploss hit or price breaks above R3 (failure of bearish breakout)
            if curr_close > entry_price + 1.5 * curr_atr or curr_close > curr_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long entry when price > 4h EMA50 (bullish regime) AND breaks above R3 with volume confirmation
            if curr_close > curr_ema50_4h and curr_high > curr_r3 and vol_confirm:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            # Short entry when price < 4h EMA50 (bearish regime) AND breaks below S3 with volume confirmation
            elif curr_close < curr_ema50_4h and curr_low < curr_s3 and vol_confirm:
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals