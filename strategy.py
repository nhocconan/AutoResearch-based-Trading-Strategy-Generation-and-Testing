#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Uses Camarilla pivot levels from prior 4h period: long on break above R3 in uptrend, short on break below S3 in downtrend
# Volume confirmation (>1.8x 20-period average) ensures institutional participation
# Trend filter uses 4h EMA50 to avoid counter-trend trades in both bull and bear markets
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods
# Designed for 1h timeframe to capture swings with controlled trade frequency (~20-35 trades/year)

name = "1h_Camarilla_R3S3_4hEMA50_VolumeConfirm_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get 4h data for EMA50 trend filter and Camarilla calculation (HTF = 4h)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla pivot levels from prior 4h bar (using previous 4h bar's OHLC)
    # Camarilla: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    camarilla_r3 = close_4h + (1.1 * (high_4h - low_4h) * 1.1 / 4)
    camarilla_s3 = close_4h - (1.1 * (high_4h - low_4h) * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe (delayed by one 4h bar for look-ahead avoidance)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Calculate 20-period average volume for confirmation (on 1h data)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 20  # volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema50_4h = ema_50_4h_aligned[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price breaks below S3 or trend turns down
            if curr_low < curr_s3 or curr_close < curr_ema50_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price breaks above R3 or trend turns up
            if curr_high > curr_r3 or curr_close > curr_ema50_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.8x 20-period average
            vol_confirm = curr_volume > 1.8 * curr_vol_ma
            
            # Long entry: price breaks above R3 in uptrend (price > EMA50)
            if vol_confirm and curr_close > curr_ema50_4h:
                if curr_high > curr_r3:  # Break above R3
                    signals[i] = 0.20
                    position = 1
                    entry_price = curr_close
            # Short entry: price breaks below S3 in downtrend (price < EMA50)
            elif vol_confirm and curr_close < curr_ema50_4h:
                if curr_low < curr_s3:  # Break below S3
                    signals[i] = -0.20
                    position = -1
                    entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals