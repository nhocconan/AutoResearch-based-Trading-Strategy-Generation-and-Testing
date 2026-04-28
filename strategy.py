#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike
# Long when price breaks above R3 (Camarilla from prior 12h bar) AND 12h EMA50 rising AND volume > 2x 20-bar avg
# Short when price breaks below S3 (Camarilla from prior 12h bar) AND 12h EMA50 falling AND volume > 2x 20-bar avg
# Exits when price retouches the Camarilla pivot point (PP) from the breakout bar
# Target: 12-37 trades/year via tight breakout conditions + trend filter reducing false breakouts
# Works in both bull and bear markets by only trading in direction of 12h trend with volume confirmation

name = "6h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla levels and EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h close
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Prepend zeros for alignment (since we lost first 49 bars in EMA calculation)
    ema_50_12h = np.concatenate([np.full(49, np.nan), ema_50_12h])
    
    # Align 12h EMA50 to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate EMA50 slope (rising/falling) on aligned array
    ema_slope = np.diff(ema_50_12h_aligned, prepend=ema_50_12h_aligned[0])
    ema_rising = ema_slope > 0
    ema_falling = ema_slope < 0
    
    # Calculate Camarilla levels for each 12h bar: based on prior 12h bar's OHLC
    # Camarilla formulas:
    # PP = (high_prev + low_prev + close_prev) / 3
    # R3 = PP + (high_prev - low_prev) * 1.1 / 4
    # S3 = PP - (high_prev - low_prev) * 1.1 / 4
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Shift by 1 to use prior bar's OHLC for current bar's levels
    high_prev = np.concatenate([[np.nan], high_12h[:-1]])
    low_prev = np.concatenate([[np.nan], low_12h[:-1]])
    close_prev = np.concatenate([[np.nan], close_12h[:-1]])
    
    pp = (high_prev + low_prev + close_prev) / 3.0
    r3 = pp + (high_prev - low_prev) * 1.1 / 4.0
    s3 = pp - (high_prev - low_prev) * 1.1 / 4.0
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    pp_aligned = align_htf_to_ltf(prices, df_12h, pp)  # for exit condition
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    breakout_pp = 0.0  # PP level from breakout bar for exit
    
    start_idx = max(50, 20)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        price = close[i]
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above R3 AND 12h EMA50 rising AND volume confirmation
            if price > r3_aligned[i] and ema_rising[i] and vol_conf:
                signals[i] = 0.25
                position = 1
                breakout_pp = pp_aligned[i]  # Store PP for exit
            # Short when price breaks below S3 AND 12h EMA50 falling AND volume confirmation
            elif price < s3_aligned[i] and ema_falling[i] and vol_conf:
                signals[i] = -0.25
                position = -1
                breakout_pp = pp_aligned[i]  # Store PP for exit
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price retouches PP or EMA50 turns falling
            if price <= pp_aligned[i] or not ema_rising[i]:
                signals[i] = 0.0
                position = 0
                breakout_pp = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price retouches PP or EMA50 turns rising
            if price >= pp_aligned[i] or not ema_falling[i]:
                signals[i] = 0.0
                position = 0
                breakout_pp = 0.0
            else:
                signals[i] = -0.25
    
    return signals