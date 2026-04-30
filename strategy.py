#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h EMA50 trend filter and volume spike confirmation.
# Long when price breaks above Camarilla R3 level with 4h uptrend (close > 4h EMA50) and volume > 2.0x 20-bar avg.
# Short when price breaks below Camarilla S3 level with 4h downtrend (close < 4h EMA50) and volume > 2.0x 20-bar avg.
# Exit on opposite Camarilla level touch (mean reversion within the pivot range).
# Uses proven Camarilla structure with strict volume confirmation (2.0x) and 4h EMA50 trend filter to limit trades (target 15-37/year).
# 4h EMA50 provides stronger trend filter than shorter EMAs for BTC/ETH, reducing false signals in choppy markets.
# Timeframe: 1h, HTF: 4h for trend alignment as per experiment guidelines.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_VolumeSpike_v1"
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
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Previous 4h OHLC for completed 4h bar (no look-ahead)
    df_4h_prev = get_htf_data(prices, '4h')
    if len(df_4h_prev) < 2:
        return np.zeros(n)
    
    prev_high_4h = df_4h_prev['high'].shift(1).values
    prev_low_4h = df_4h_prev['low'].shift(1).values
    prev_close_4h = df_4h_prev['close'].shift(1).values
    
    # Align 4h data to 1h timeframe (completed 4h bar only)
    prev_high_aligned = align_htf_to_ltf(prices, df_4h_prev, prev_high_4h)
    prev_low_aligned = align_htf_to_ltf(prices, df_4h_prev, prev_low_4h)
    prev_close_aligned = align_htf_to_ltf(prices, df_4h_prev, prev_close_4h)
    
    # Calculate Camarilla levels from previous completed 4h bar (no look-ahead)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.25*(high-low), etc.
    camarilla_high = prev_close_aligned + 1.25 * (prev_high_aligned - prev_low_aligned)  # R3
    camarilla_low = prev_close_aligned - 1.25 * (prev_high_aligned - prev_low_aligned)   # S3
    
    # Volume confirmation: volume > 2.0x 20-period average (strict to avoid overtrading)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 and Camarilla
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(camarilla_high[i]) or np.isnan(camarilla_low[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_camarilla_high = camarilla_high[i]
        curr_camarilla_low = camarilla_low[i]
        curr_ema_50_4h = ema_50_4h_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3, uptrend (close > 4h EMA50), volume spike
            if (curr_close > curr_camarilla_high and 
                curr_close > curr_ema_50_4h and 
                curr_volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla S3, downtrend (close < 4h EMA50), volume spike
            elif (curr_close < curr_camarilla_low and 
                  curr_close < curr_ema_50_4h and 
                  curr_volume_confirm):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price touches Camarilla S3 (mean reversion)
            if curr_close <= curr_camarilla_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit condition: price touches Camarilla R3 (mean reversion)
            if curr_close >= curr_camarilla_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals