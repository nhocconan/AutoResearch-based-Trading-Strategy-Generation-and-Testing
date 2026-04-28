#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume spike confirmation.
# Enter long when price breaks above 12h Camarilla R3 level with 1w EMA50 uptrend and volume > 2.0x 20-bar average.
# Enter short when price breaks below 12h Camarilla S3 level with 1w EMA50 downtrend and volume confirmation.
# Exit when price retraces to the 12h Camarilla midpoint (R3/S3 average).
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 50-150 total trades over 4 years (12-37/year).
# Camarilla pivot levels provide strong support/resistance, especially on 12h timeframe.
# Weekly EMA50 filter ensures we only trade with the major trend, avoiding counter-trend whipsaws.
# Volume confirmation filters weak breakouts. This combination has shown promise on lower timeframes
# and should work on 12h with proper trade frequency for BTC/ETH.

name = "12h_Camarilla_R3S3_Breakout_1wEMA50_Trend_VolumeSpike_v1"
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
    
    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 2:  # Need at least 2 periods for pivot calculation
        return np.zeros(n)
    
    # Calculate 12h Camarilla pivot levels (R3, S3, midpoint)
    # Camarilla formulas based on previous bar's OHLC
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate pivot point (PP) = (H + L + C) / 3
    pp = (high_12h + low_12h + close_12h) / 3.0
    # Calculate range
    range_12h = high_12h - low_12h
    # Camarilla R3 = C + (H-L) * 1.1/4
    camarilla_r3 = close_12h + range_12h * 1.1 / 4.0
    # Camarilla S3 = C - (H-L) * 1.1/4
    camarilla_s3 = close_12h - range_12h * 1.1 / 4.0
    # Camarilla midpoint (for exit) = (R3 + S3) / 2
    camarilla_mid = (camarilla_r3 + camarilla_s3) / 2.0
    
    # Align Camarilla levels to 12h
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_12h, camarilla_mid)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:  # Need sufficient data for EMA calculation
        return np.zeros(n)
    
    # Calculate 1w EMA (50-period)
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA to 12h
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(2, 50)  # Ensure sufficient history for pivots and EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_mid_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1w EMA trend filter: price > EMA50 = uptrend, price < EMA50 = downtrend
        ema_trend_up = close[i] > ema_50_aligned[i]
        ema_trend_down = close[i] < ema_50_aligned[i]
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price > Camarilla R3, price > EMA50 (uptrend), volume confirm
            if price > camarilla_r3_aligned[i] and ema_trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price < Camarilla S3, price < EMA50 (downtrend), volume confirm
            elif price < camarilla_s3_aligned[i] and ema_trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit at midpoint
            if price <= camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - hold or exit at midpoint
            if price >= camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals