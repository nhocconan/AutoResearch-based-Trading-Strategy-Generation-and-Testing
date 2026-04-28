#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above R3 AND close > 1w EMA50 AND volume > 2.0x 20-bar avg
# Short when price breaks below S3 AND close < 1w EMA50 AND volume > 2.0x 20-bar avg
# Exits when price retouches the opposite Camarilla level (S3 for longs, R3 for shorts)
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 7-25 trades/year on 1d.
# Works in bull markets by trading breakouts with trend, works in bear by requiring volume spikes
# which often accompany panic selling/buying climaxes that precede reversals.

name = "1d_Camarilla_R3S3_Breakout_1wEMA50_Trend_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 and Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA(50) on 1w close
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels from previous 1w OHLC
    # We need the OHLC of the completed 1w bar to calculate levels for current period
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels: R3, S3
    # R3 = close + ((high - low) * 1.1/4)
    # S3 = close - ((high - low) * 1.1/4)
    camarilla_r3 = close_1w + ((high_1w - low_1w) * 1.1 / 4)
    camarilla_s3 = close_1w - ((high_1w - low_1w) * 1.1 / 4)
    
    # Align 1w EMA50 to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Align 1w Camarilla levels to 1d timeframe (use completed 1w bar's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # Volume confirmation: >2.0x 20-bar average volume (strict filter to reduce trades)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_trend = ema_50_1w_aligned[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        curr_close = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above R3 AND close > 1w EMA50 AND volume confirmation
            if curr_close > r3_level and curr_close > ema_trend and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S3 AND close < 1w EMA50 AND volume confirmation
            elif curr_close < s3_level and curr_close < ema_trend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price retouches S3 (opposite level)
            if curr_close <= s3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price retouches R3 (opposite level)
            if curr_close >= r3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals