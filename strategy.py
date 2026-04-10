#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla breakout with 4h trend filter and 1d volume confirmation
# - Long when price breaks above 1h Camarilla R3 level AND 4h close > 4h EMA50 AND 1d volume > 1.3x 20-period volume SMA
# - Short when price breaks below 1h Camarilla S3 level AND 4h close < 4h EMA50 AND 1d volume > 1.3x 20-period volume SMA
# - Exit: price retreats to Camarilla pivot point (PP) OR loss of volume confirmation
# - Position sizing: 0.20 discrete level to minimize fee drag
# - Session filter: 08-20 UTC to reduce noise
# - Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe
# - Uses 4h for trend direction, 1d for volume confirmation, 1h for entry timing

name = "1h_4h_1d_camarilla_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 1h Camarilla pivot levels from 1d OHLC (structure from higher timeframe)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_pp = (high_1d + low_1d + close_1d) / 3.0
    camarilla_range = high_1d - low_1d
    camarilla_r3 = camarilla_pp + camarilla_range * 1.1 / 4.0
    camarilla_s3 = camarilla_pp - camarilla_range * 1.1 / 4.0
    
    # Align 1d Camarilla levels to 1h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    for i in range(50, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC only
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.3x 20-period volume SMA
        # Need to map 1h index to 1d index: each 1d bar = 24 1h bars
        vol_1d_idx = i // 24
        if vol_1d_idx < len(volume_1d):
            vol_confirm = volume_1d[vol_1d_idx] > 1.3 * volume_sma_20_1d_aligned[i]
        else:
            vol_confirm = False
        
        # Trend filter: 4h close vs 4h EMA50
        trend_bullish = close[i] > ema_50_4h_aligned[i]  # Using 1h close vs aligned 4h EMA (approximation)
        trend_bearish = close[i] < ema_50_4h_aligned[i]
        
        # Camarilla breakout signals (using 1h price vs prior aligned levels)
        breakout_up = close[i] > camarilla_r3_aligned[i-1]  # Break above previous R3
        breakout_down = close[i] < camarilla_s3_aligned[i-1]  # Break below previous S3
        
        # Exit conditions: price retreats to pivot point or loss of volume confirmation
        exit_long = close[i] < camarilla_pp_aligned[i] or not vol_confirm
        exit_short = close[i] > camarilla_pp_aligned[i] or not vol_confirm
        
        if position == 0:  # Flat - look for entry
            if breakout_up and trend_bullish and vol_confirm:
                position = 1
                signals[i] = 0.20
            elif breakout_down and trend_bearish and vol_confirm:
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
    
    return signals