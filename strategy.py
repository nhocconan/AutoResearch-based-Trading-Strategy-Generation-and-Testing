#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 with 1w uptrend (price > 1w EMA34) and volume spike (>1.6x 20-bar avg).
# Short when price breaks below Camarilla S3 with 1w downtrend (price < 1w EMA34) and volume spike.
# Exit on touch of Camarilla H3/L3 levels (mean reversion within the inner range).
# Uses proven Camarilla structure with tight volume confirmation to limit trades (target 30-100 total trades over 4 years).
# 1d timeframe reduces fee drag while capturing medium-term swings; 1w EMA filter ensures alignment with major trend.

name = "1d_Camarilla_R3S3_Breakout_1wEMA34_Trend_VolumeSpike_v1"
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
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Previous 1w OHLC for completed 1w bar (no look-ahead)
    df_1w_prev = get_htf_data(prices, '1w')
    if len(df_1w_prev) < 2:
        return np.zeros(n)
    
    prev_high_1w = df_1w_prev['high'].shift(1).values
    prev_low_1w = df_1w_prev['low'].shift(1).values
    prev_close_1w = df_1w_prev['close'].shift(1).values
    
    # Align 1w data to 1d timeframe (completed 1w bar only)
    prev_high_aligned = align_htf_to_ltf(prices, df_1w_prev, prev_high_1w)
    prev_low_aligned = align_htf_to_ltf(prices, df_1w_prev, prev_low_1w)
    prev_close_aligned = align_htf_to_ltf(prices, df_1w_prev, prev_close_1w)
    
    # Camarilla levels from previous completed 1w bar (no look-ahead)
    # Camarilla: H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2
    #          H3 = C + 1.1*(H-L)/4, L3 = C - 1.1*(H-L)/4
    #          H2 = C + 1.1*(H-L)/6, L2 = C - 1.1*(H-L)/6
    #          H1 = C + 1.1*(H-L)/12, L1 = C - 1.1*(H-L)/12
    #          PP = (H+L+C)/3
    # We use R3=H3 and S3=L3 for breakouts, H3/L3 for exits
    rng = prev_high_aligned - prev_low_aligned
    camarilla_pp = (prev_high_aligned + prev_low_aligned + prev_close_aligned) / 3.0
    camarilla_h3 = camarilla_pp + 1.1 * rng / 4.0
    camarilla_l3 = camarilla_pp - 1.1 * rng / 4.0
    camarilla_h4 = camarilla_pp + 1.1 * rng / 2.0
    camarilla_l4 = camarilla_pp - 1.1 * rng / 2.0
    
    # Volume confirmation: volume > 1.6x 20-period average (tight to avoid overtrading)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.6 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA34 and Camarilla
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_camarilla_h3 = camarilla_h3[i]
        curr_camarilla_l3 = camarilla_l3[i]
        curr_camarilla_h4 = camarilla_h4[i]
        curr_camarilla_l4 = camarilla_l4[i]
        curr_ema_34_1w = ema_34_1w_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla H3, uptrend (price > 1w EMA34), volume spike
            if (curr_close > curr_camarilla_h3 and 
                curr_close > curr_ema_34_1w and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla L3, downtrend (price < 1w EMA34), volume spike
            elif (curr_close < curr_camarilla_l3 and 
                  curr_close < curr_ema_34_1w and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price touches Camarilla L3 (mean reversion to midpoint)
            if curr_close <= curr_camarilla_l3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price touches Camarilla H3 (mean reversion to midpoint)
            if curr_close >= curr_camarilla_h3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals