#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Camarilla Pivot Breakout + 1d EMA34 Trend + Volume Spike
# Weekly Camarilla levels act as strong support/resistance. Breakouts above R4 or below S4
# indicate strong momentum. Filtered by 1d EMA34 trend (long above, short below) and
# volume confirmation (>2.0x 20-bar MA). Works in bull/bear via trend alignment.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing (0.25).

name = "6h_WeeklyCamarilla_R4S4_Breakout_1dEMA34_Trend_VolumeConfirm_v1"
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
    
    # Weekly HTF data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels (using previous week's OHLC)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    camarilla_r4 = weekly_close + (weekly_high - weekly_low) * 1.50
    camarilla_s4 = weekly_close - (weekly_high - weekly_low) * 1.50
    
    # Align weekly Camarilla levels to 6h timeframe (no additional delay needed for pivot levels)
    camarilla_r4_6h = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_s4_6h = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
    # Daily HTF data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Daily EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 20  # Need 20 for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(camarilla_r4_6h[i]) or np.isnan(camarilla_s4_6h[i]) or np.isnan(ema_34_6h[i]) or np.isnan(volume_ma_20[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Close above weekly R4, above daily EMA34, and volume confirmation
            if curr_close > camarilla_r4_6h[i] and curr_close > ema_34_6h[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Close below weekly S4, below daily EMA34, and volume confirmation
            elif curr_close < camarilla_s4_6h[i] and curr_close < ema_34_6h[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on close below weekly R4 or below daily EMA34
            if curr_close < camarilla_r4_6h[i] or curr_close < ema_34_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on close above weekly S4 or above daily EMA34
            if curr_close > camarilla_s4_6h[i] or curr_close > ema_34_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals