#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation.
# Long: Close breaks above Camarilla R3 AND 1w EMA34 trending up AND volume > 1.5x 20-period MA
# Short: Close breaks below Camarilla S3 AND 1w EMA34 trending down AND volume > 1.5x 20-period MA
# Exit: Opposite Camarilla breakout or EMA trend reversal or volume drops.
# Discrete sizing 0.25. Target: 50-150 total trades over 4 years (12-37/year).
# Camarilla provides precise pivot levels; 1w EMA34 filters for strong weekly trends; volume confirmation
# reduces false breakouts. Works in bull via long signals and bear via short signals when aligned with weekly trend.

name = "12h_Camarilla_R3S3_1wEMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # Calculate 1w EMA34
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA34 to 12h timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d data for Camarilla pivot levels (using previous 1d bar)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1d bar: based on previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3 and S3 levels
    # R3 = close + 1.1*(high - low)/2
    # S3 = close - 1.1*(high - low)/2
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 12h timeframe (using previous completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume regime: current 12h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_val = ema_34_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine EMA trend (using previous bar's EMA for trend direction)
        if i > 100:
            ema_prev = ema_34_1w_aligned[i-1]
            ema_trend_up = ema_val > ema_prev
            ema_trend_down = ema_val < ema_prev
        else:
            ema_trend_up = True
            ema_trend_down = False
        
        # Entry logic
        if position == 0:
            # Long: Close breaks above Camarilla R3 AND EMA trending up AND volume spike
            if close_val > camarilla_r3_aligned[i] and ema_trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Camarilla S3 AND EMA trending down AND volume spike
            elif close_val < camarilla_s3_aligned[i] and ema_trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close breaks below Camarilla S3 OR EMA trend reverses down OR volume drops
            if close_val < camarilla_s3_aligned[i] or not ema_trend_up or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close breaks above Camarilla R3 OR EMA trend reverses up OR volume drops
            if close_val > camarilla_r3_aligned[i] or not ema_trend_down or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals