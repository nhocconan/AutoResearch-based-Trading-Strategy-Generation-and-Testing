#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d trend filter (EMA34) and volume confirmation (>1.5x 20-bar MA)
# Uses 1d EMA for stronger trend alignment, reducing whipsaws in ranging markets.
# Camarilla R3/S3 levels act as magnet points - breaks indicate strong momentum.
# Volume confirmation ensures participation and reduces false breakouts.
# Discrete sizing (0.25) minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year) with strong BTC/ETH performance.

name = "6h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeConfirm_v1"
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
    open_price = prices['open'].values
    
    # 1d HTF data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA(34) on 1d close
    ema_1d_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 6h timeframe
    ema_1d_34_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_34)
    
    # Calculate Camarilla levels from previous 1d bar (using typical price)
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    
    # Use previous day's typical price for Camarilla calculation
    prev_typical = pd.Series(typical_price).shift(1).values
    prev_high = pd.Series(high).shift(1).values
    prev_low = pd.Series(low).shift(1).values
    prev_close = pd.Series(close).shift(1).values
    
    # Camarilla levels calculation
    # R3 = close + 1.1*(high-low)/2
    # S3 = close - 1.1*(high-low)/2
    # R4 = close + 1.1*(high-low)
    # S4 = close - 1.1*(high-low)
    cam_r3 = prev_close + 1.1 * (prev_high - prev_low) / 2.0
    cam_s3 = prev_close - 1.1 * (prev_high - prev_low) / 2.0
    cam_r4 = prev_close + 1.1 * (prev_high - prev_low)
    cam_s4 = prev_close - 1.1 * (prev_high - prev_low)
    
    # Align Camarilla levels to 6h timeframe
    cam_r3_aligned = align_htf_to_ltf(prices, df_1d, cam_r3)
    cam_s3_aligned = align_htf_to_ltf(prices, df_1d, cam_s3)
    cam_r4_aligned = align_htf_to_ltf(prices, df_1d, cam_r4)
    cam_s4_aligned = align_htf_to_ltf(prices, df_1d, cam_s4)
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 20  # Need 20 for volume MA and 1 for shift
    
    for i in range(start_idx, n):
        if np.isnan(ema_1d_34_aligned[i]) or np.isnan(cam_r3_aligned[i]) or np.isnan(cam_s3_aligned[i]) or np.isnan(volume_ma_20[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3, above 1d EMA, and volume confirmation
            if curr_high > cam_r3_aligned[i] and curr_close > ema_1d_34_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3, below 1d EMA, and volume confirmation
            elif curr_low < cam_s3_aligned[i] and curr_close < ema_1d_34_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on price breaking below Camarilla S3 or below 1d EMA
            if curr_low < cam_s3_aligned[i] or curr_close < ema_1d_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on price breaking above Camarilla R3 or above 1d EMA
            if curr_high > cam_r3_aligned[i] or curr_close > ema_1d_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals