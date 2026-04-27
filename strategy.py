# 1. Hypothesis:
# Strategy combines daily Camarilla pivot reversals (S3/R3) with weekly trend filter (EMA34)
# and volume confirmation to capture mean-reversion bounces in established trends.
# Works in bull markets by buying pullbacks to S3 in uptrends, and in bear markets by
# selling rallies to R3 in downtrends. Volume confirmation reduces false signals.
# Uses 12h timeframe for balance between signal quality and trade frequency.
# Target: 20-40 trades/year per symbol to avoid fee drag.

# 2. Implementation:
# - Daily Camarilla S3/R3 levels calculated from prior day's HLC
# - Weekly EMA(34) for trend filter
# - Volume > 1.8x 20-period average for confirmation
# - Entry: Price touches/passes S3/R3 with volume and weekly trend alignment
# - Exit: Price reaches opposite level (R3 for longs, S3 for shorts) or trend reverses
# - Position size: 0.25 (25% of capital) to balance risk and return

# 3. Risk Management:
# - Weekly trend filter prevents counter-trend trading in strong moves
# - Volume confirmation ensures institutional participation
# - Fixed position size avoids over-leveraging
# - Exit on trend change or opposite level hit limits losses

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot levels (high, low, close of previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day using previous day's HLC
    # R3 = C + ((H-L)*1.1/4)
    # S3 = C - ((H-L)*1.1/4)
    camarilla_r3 = np.zeros(len(df_1d))
    camarilla_s3 = np.zeros(len(df_1d))
    
    for i in range(1, len(df_1d)):
        h = high_1d[i-1]
        l = low_1d[i-1]
        c = close_1d[i-1]
        range_hl = h - l
        camarilla_r3[i] = c + (range_hl * 1.1 / 4)
        camarilla_s3[i] = c - (range_hl * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get weekly trend filter: EMA(34) on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w_34 = np.full(len(df_1w), np.nan)
    # Use pandas EMA for accuracy and simplicity
    ema_series = pd.Series(close_1w).ewm(span=34, adjust=False).mean()
    ema_1w_34 = ema_series.values
    
    ema_1w_34_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_34)
    
    # Calculate volume average (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(20, 34)  # volume MA needs 20, weekly EMA needs 34
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_1w_34_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        
        # Volume confirmation: > 1.8x average volume (strict to reduce trades)
        volume_confirmation = vol_ratio > 1.8
        
        if position == 0:
            # Long: price touches or breaks above S3 with volume and weekly uptrend
            if (volume_confirmation and 
                price >= camarilla_s3_aligned[i] and 
                close[i-1] < camarilla_s3_aligned[i] and  # just touched/broke
                ema_1w_34_aligned[i] > ema_1w_34_aligned[i-1]):  # weekly uptrend
                signals[i] = 0.25
                position = 1
            # Short: price touches or breaks below R3 with volume and weekly downtrend
            elif (volume_confirmation and 
                  price <= camarilla_r3_aligned[i] and 
                  close[i-1] > camarilla_r3_aligned[i] and  # just touched/broke
                  ema_1w_34_aligned[i] < ema_1w_34_aligned[i-1]):  # weekly downtrend
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price reaches R3 or weekly trend turns down
            if (price >= camarilla_r3_aligned[i] or 
                ema_1w_34_aligned[i] < ema_1w_34_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price reaches S3 or weekly trend turns up
            if (price <= camarilla_s3_aligned[i] or 
                ema_1w_34_aligned[i] > ema_1w_34_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "12h_Camarilla_S3R3_WeeklyEMA34_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0