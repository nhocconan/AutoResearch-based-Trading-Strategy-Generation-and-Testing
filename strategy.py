#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 with 1d uptrend (close > 1d EMA34) and volume > 2.0x 20-bar avg.
# Short when price breaks below Camarilla S3 with 1d downtrend (close < 1d EMA34) and volume > 2.0x 20-bar avg.
# Exit on touch of Camarilla H3/L3 levels (mean reversion within the inner range).
# Uses proven Camarilla structure with strict volume confirmation and 1d EMA34 trend filter to limit trades (target 12-37/year).
# 1d EMA34 provides longer-term trend filter, reducing false signals in choppy markets and bear rallies.
# Timeframe: 6h, HTF: 1d as per experiment guidelines.

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
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
    
    # Load 1d data ONCE before loop for trend filter and pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Previous 1d OHLC for completed 1d bar (no look-ahead)
    df_1d_prev = get_htf_data(prices, '1d')
    if len(df_1d_prev) < 2:
        return np.zeros(n)
    
    prev_high_1d = df_1d_prev['high'].shift(1).values
    prev_low_1d = df_1d_prev['low'].shift(1).values
    prev_close_1d = df_1d_prev['close'].shift(1).values
    
    # Align 1d data to 6h timeframe (completed 1d bar only)
    prev_high_aligned = align_htf_to_ltf(prices, df_1d_prev, prev_high_1d)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d_prev, prev_low_1d)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d_prev, prev_close_1d)
    
    # Calculate Camarilla pivot levels from previous completed 1d bar
    # Camarilla: H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
    #          H4 = C + (H-L)*1.1/2, L4 = C - (H-L)*1.1/2
    #          R3 = H3, S3 = L3 (using H3/L3 as R3/S3 for breakout)
    #          R4 = H4, S4 = L4 (using H4/L4 as R4/S4 for continuation)
    prev_range = prev_high_aligned - prev_low_aligned
    camarilla_h3 = prev_close_aligned + prev_range * 1.1 / 4
    camarilla_l3 = prev_close_aligned - prev_range * 1.1 / 4
    camarilla_h4 = prev_close_aligned + prev_range * 1.1 / 2
    camarilla_l4 = prev_close_aligned - prev_range * 1.1 / 2
    
    # Volume confirmation: volume > 2.0x 20-period average (strict to avoid overtrading)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or
            np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla H3 (R3), uptrend (close > 1d EMA34), volume spike
            if (curr_close > camarilla_h3[i] and 
                curr_close > curr_ema_34_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla L3 (S3), downtrend (close < 1d EMA34), volume spike
            elif (curr_close < camarilla_l3[i] and 
                  curr_close < curr_ema_34_1d and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price touches Camarilla L3 (mean reversion)
            if curr_close <= camarilla_l3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price touches Camarilla H3 (mean reversion)
            if curr_close >= camarilla_h3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals