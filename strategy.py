#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike (>2x 20-bar MA)
# Camarilla R3/S3 levels act as strong intraday support/resistance; breakouts with volume and trend alignment capture momentum.
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue) when aligned with higher timeframe trend.
# Discrete sizing (0.25) minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA(34) on 1d close
    ema_1d_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 6h timeframe
    ema_1d_34_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_34)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Need to align 1d OHLC to 6s timeframe
    df_1d_ohlc = get_htf_data(prices, '1d')[['open', 'high', 'low', 'close']]
    if len(df_1d_ohlc) < 1:
        return np.zeros(n)
    
    # Get previous day's OHLC for Camarilla calculation
    prev_close = df_1d_ohlc['close'].shift(1).values
    prev_high = df_1d_ohlc['high'].shift(1).values
    prev_low = df_1d_ohlc['low'].shift(1).values
    prev_open = df_1d_ohlc['open'].shift(1).values
    
    # Align 1d OHLC to 6h timeframe (wait for day to close)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d_ohlc, prev_close)
    prev_high_aligned = align_htf_to_ltf(prices, df_1d_ohlc, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d_ohlc, prev_low)
    prev_open_aligned = align_htf_to_ltf(prices, df_1d_ohlc, prev_open)
    
    # Calculate Camarilla levels
    # R4 = Close + ((High-Low) * 1.1/2)
    # R3 = Close + ((High-Low) * 1.1/4)
    # S3 = Close - ((High-Low) * 1.1/4)
    # S4 = Close - ((High-Low) * 1.1/2)
    diff = prev_high_aligned - prev_low_aligned
    r3 = prev_close_aligned + (diff * 1.1 / 4)
    s3 = prev_close_aligned - (diff * 1.1 / 4)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 20  # Need 20 for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_34_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above R3, above 1d EMA34, volume spike
            if curr_close > r3[i] and curr_close > ema_1d_34_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3, below 1d EMA34, volume spike
            elif curr_close < s3[i] and curr_close < ema_1d_34_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on price below S3 or below 1d EMA34
            if curr_close < s3[i] or curr_close < ema_1d_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on price above R3 or above 1d EMA34
            if curr_close > r3[i] or curr_close > ema_1d_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals