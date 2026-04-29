#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla Pivot R3/S3 breakout with 1w EMA50 trend filter and volume spike confirmation
# Long when price breaks above R3 AND close > 1w EMA50 AND volume > 2.0x 20-bar avg
# Short when price breaks below S3 AND close < 1w EMA50 AND volume > 2.0x 20-bar avg
# Exit when price reverts to opposite Camarilla level (long exit at S3, short exit at R3)
# Uses discrete position sizing (0.30) to minimize fee drag. Target: 7-25 trades/year on 1d.
# Camarilla pivots provide precise intraday support/resistance levels that work in both trending and ranging markets.
# 1w EMA50 filter ensures alignment with higher timeframe trend for better win rate.
# Volume confirmation ensures signals have conviction, reducing false breakouts.

name = "1d_Camarilla_R3S3_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA(50) on 1w close
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla pivots from previous day
    # Camarilla levels: 
    # H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    # H3 = close + 1.25*(high-low), L3 = close - 1.25*(high-low)
    # H2 = close + 1.166*(high-low), L2 = close - 1.166*(high-low)
    # H1 = close + 1.0833*(high-low), L1 = close - 1.0833*(high-low)
    # We focus on R3 (H3) and S3 (L3) for breakouts
    
    # Use previous day's OHLC to calculate today's pivots
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    
    # First bar has no previous data
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Calculate Camarilla R3 and S3 levels
    diff = prev_high - prev_low
    r3 = prev_close + 1.25 * diff  # Resistance level 3
    s3 = prev_close - 1.25 * diff  # Support level 3
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Need sufficient history for volume MA and EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_trend = ema_50_1w_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above R3 AND close > 1w EMA50 AND volume confirmation
            if curr_high > r3[i] and curr_close > ema_trend and vol_conf:
                signals[i] = 0.30
                position = 1
            # Short when price breaks below S3 AND close < 1w EMA50 AND volume confirmation
            elif curr_low < s3[i] and curr_close < ema_trend and vol_conf:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price breaks below S3 (reversion to mean)
            if curr_low < s3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:  # Short - exit when price breaks above R3 (reversion to mean)
            if curr_high > r3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals