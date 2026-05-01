#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA(34) trend filter and volume confirmation (>1.5x 20-bar MA)
# Camarilla pivot levels provide precise daily support/resistance. Breakout above R3 or below S3 with
# 1w EMA trend alignment and volume confirmation captures strong momentum moves. Works in bull markets
# via breakouts above R3 with uptrend, and in bear markets via breakdowns below S3 with downtrend.
# Target: 30-100 total trades over 4 years (7-25/year) with discrete sizing (0.30).

name = "1d_Camarilla_R3S3_Breakout_1wEMA34_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for EMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA(34) on 1w close
    weekly_ema_34 = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly EMA to 1d timeframe
    weekly_ema_34_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema_34)
    
    # Camarilla pivot levels from 1d data (using previous day's OHLC)
    # Camarilla levels: R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    prev_high = prices['high'].shift(1).values
    prev_low = prices['low'].shift(1).values
    prev_close = prices['close'].shift(1).values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low)
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(20, 34)  # Need 20 for volume MA, 34 for EMA
    
    for i in range(start_idx, n):
        if np.isnan(weekly_ema_34_aligned[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or np.isnan(volume_ma_20[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3, above weekly EMA, and volume confirmation
            if curr_close > camarilla_r3[i] and curr_close > weekly_ema_34_aligned[i] and vol_confirm:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Camarilla S3, below weekly EMA, and volume confirmation
            elif curr_close < camarilla_s3[i] and curr_close < weekly_ema_34_aligned[i] and vol_confirm:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on price breaking below Camarilla S3 or below weekly EMA
            if curr_close < camarilla_s3[i] or curr_close < weekly_ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit on price breaking above Camarilla R3 or above weekly EMA
            if curr_close > camarilla_r3[i] or curr_close > weekly_ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals