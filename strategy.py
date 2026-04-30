#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike (>2.0x average)
# Uses 4h timeframe to balance trade frequency (target: 75-200 total trades over 4 years)
# 1d EMA34 provides trend filter for bull/bear markets (avoid counter-trend trades)
# Volume confirmation >2.0x 20-period average reduces false breakouts
# Discrete position sizing: 0.30 for entries to limit fee drag
# Works in all regimes: breakouts occur in all markets, volume confirms legitimacy, trend filter avoids counter-trend

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Camarilla levels from previous day (using 1d OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Camarilla levels: based on previous day's range
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    range_ = prev_high - prev_low
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = prev_close + (range_ * 1.1 / 4)
    camarilla_s3 = prev_close - (range_ * 1.1 / 4)
    
    # Align to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 34, 20)  # warmup for EMA (34), volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on Camarilla level break with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish breakout: price above R3 + above 1d EMA34
                if curr_close > curr_r3 and curr_close > curr_ema_34_1d:
                    signals[i] = 0.30
                    position = 1
                # Bearish breakout: price below S3 + below 1d EMA34
                elif curr_close < curr_s3 and curr_close < curr_ema_34_1d:
                    signals[i] = -0.30
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: price closes below S3 (reversal) or above R3 (take profit)
            if curr_close < curr_s3 or curr_close > curr_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: price closes above R3 (reversal) or below S3 (take profit)
            if curr_close > curr_r3 or curr_close < curr_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals