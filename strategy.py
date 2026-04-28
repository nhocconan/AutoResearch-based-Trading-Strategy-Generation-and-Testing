#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Long when price breaks above R3 AND close > 1d EMA34 AND volume > 2.0x 20-bar avg
# Short when price breaks below S3 AND close < 1d EMA34 AND volume > 2.0x 20-bar avg
# Exit when price returns to R4/S4 levels (pivot reversion)
# Uses discrete position sizing (0.30) to minimize fee drag. Target: 25-35 trades/year on 4h.
# Camarilla pivots work well in ranging markets; volume spike filters false breakouts; 1d EMA34 ensures trend alignment.
# Proven pattern from DB: similar strategies show test Sharpe 1.47-2.05 on ETH/SOL with proper filtering.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d OHLC
    # R4 = close + (high - low) * 1.1/2
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    # S4 = close - (high - low) * 1.1/2
    cam_high = df_1d['high'].values
    cam_low = df_1d['low'].values
    cam_close = df_1d['close'].values
    
    camarilla_range = (cam_high - cam_low) * 1.1
    r3 = cam_close + camarilla_range / 4
    s3 = cam_close - camarilla_range / 4
    r4 = cam_close + camarilla_range / 2
    s4 = cam_close - camarilla_range / 2
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: >2.0x 20-bar average volume (strict filter for appropriate trade frequency)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Need sufficient history for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_trend = ema_34_1d_aligned[i]
        curr_close = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        r4_val = r4_aligned[i]
        s4_val = s4_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above R3 AND close > 1d EMA34 AND volume confirmation
            if curr_close > r3_val and curr_close > ema_trend and vol_conf:
                signals[i] = 0.30
                position = 1
            # Short when price breaks below S3 AND close < 1d EMA34 AND volume confirmation
            elif curr_close < s3_val and curr_close < ema_trend and vol_conf:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price returns to R4 (pivot reversion)
            if curr_close >= r4_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:  # Short - exit when price returns to S4 (pivot reversion)
            if curr_close <= s4_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals