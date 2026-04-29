#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Camarilla levels provide precise intraday support/resistance from 1d OHLC
# Breakouts at R3/S3 with volume >2.0x confirm institutional participation
# 1d EMA34 ensures alignment with daily trend to avoid counter-trend whipsaws
# Discrete sizing (0.25) minimizes fee churn; target 50-150 total trades over 4 years

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla levels (R3, S3) from prior 1d OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior 1d OHLC (shifted by 1 to avoid look-ahead)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculations
    camarilla_high = high_1d[-1]  # Use most recent completed 1d candle
    camarilla_low = low_1d[-1]
    camarilla_close = close_1d[-1]
    camarilla_range = camarilla_high - camarilla_low
    
    # Camarilla R3 and S3 levels
    r3 = camarilla_close + camarilla_range * 1.1 / 4
    s3 = camarilla_close - camarilla_range * 1.1 / 4
    
    # Expand to full length for alignment
    r3_full = np.full(len(df_1d), r3)
    s3_full = np.full(len(df_1d), s3)
    
    # Align to 6h timeframe (wait for completed 1d bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_full)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_full)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # warmup for volume and EMA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_volume_confirm = volume_confirm[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_ema_34 = ema_34_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish entry: price breaks above R3 + above 1d EMA34
                if curr_close > curr_r3 and curr_close > curr_ema_34:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below S3 + below 1d EMA34
                elif curr_close < curr_s3 and curr_close < curr_ema_34:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: price breaks below S3 (mean reversion) or reverses below EMA34
            if curr_close < curr_s3 or curr_close < curr_ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above R3 (mean reversion) or reverses above EMA34
            if curr_close > curr_r3 or curr_close > curr_ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals