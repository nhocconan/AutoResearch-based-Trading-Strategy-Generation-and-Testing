#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation
# Uses tight Camarilla levels (R3/S3) from weekly data for high-probability breakouts
# 1w EMA50 provides strong trend filter to avoid counter-trend trades in BTC/ETH
# Volume > 1.5x average confirms participation and reduces false breakouts
# Discrete position sizing (0.25) with Camarilla R4/S4 mean reversion exit
# Designed for ~15-30 trades/year to minimize fee drag while capturing strong moves
# Works in bull/bear via trend filter - only trades in direction of 1w EMA50

name = "12h_Camarilla_R3S3_1wEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter and Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1w Camarilla pivot levels (based on previous week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla calculations based on previous week
    # Pivot point = (H + L + C) / 3
    pp = (high_1w + low_1w + close_1w) / 3.0
    # R3 = C + (H - L) * 1.1 / 4
    r3 = close_1w + (high_1w - low_1w) * 1.1 / 4.0
    # S3 = C - (H - L) * 1.1 / 4
    s3 = close_1w - (high_1w - low_1w) * 1.1 / 4.0
    # R4 = C + (H - L) * 1.1 / 2
    r4 = close_1w + (high_1w - low_1w) * 1.1 / 2.0
    # S4 = C - (H - L) * 1.1 / 2
    s4 = close_1w - (high_1w - low_1w) * 1.1 / 2.0
    
    # Use previous week's values (shift by 1) to avoid look-ahead
    pp_shifted = np.roll(pp, 1)
    r3_shifted = np.roll(r3, 1)
    s3_shifted = np.roll(s3, 1)
    r4_shifted = np.roll(r4, 1)
    s4_shifted = np.roll(s4, 1)
    pp_shifted[0] = np.nan
    r3_shifted[0] = np.nan
    s3_shifted[0] = np.nan
    r4_shifted[0] = np.nan
    s4_shifted[0] = np.nan
    
    # Align 1w indicators to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp_shifted)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3_shifted)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3_shifted)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4_shifted)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4_shifted)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Volume and 1w EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_pp = pp_aligned[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_r4 = r4_aligned[i]
        curr_s4 = s4_aligned[i]
        curr_ema50_1w = ema_50_1w_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price below R4 (mean reversion to R4 level)
            if curr_close < curr_r4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above S4 (mean reversion to S4 level)
            if curr_close > curr_s4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirmed = curr_volume > 1.5 * curr_vol_ma
            
            # Long when price breaks above R3, 1w EMA50 up-trend, volume confirmed
            if curr_high > curr_r3 and curr_close > curr_ema50_1w and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S3, 1w EMA50 down-trend, volume confirmed
            elif curr_low < curr_s3 and curr_close < curr_ema50_1w and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals