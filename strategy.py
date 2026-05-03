#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike.
# Uses actual Camarilla pivot levels from prior 1d bar (H, L, C) to derive R3/S3 levels.
# Long when price breaks above R3 with volume > 1.5x 20-period MA and close > 1d EMA34 (uptrend).
# Short when price breaks below S3 with volume spike and close < 1d EMA34 (downtrend).
# Discrete sizing 0.25. Target: 75-200 total trades over 4 years (19-50/year).
# Camarilla levels provide structure; EMA34 filters counter-trend trades.
# Volume confirmation reduces false breakouts. Works in bull/bear via trend alignment.

name = "4h_Camarilla_R3S3_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 1d data for Camarilla pivot levels (R3, S3)
    # Camarilla: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # Using prior completed 1d bar (shift 1)
    camarilla_high = pd.Series(df_1d['high'].values).shift(1).values
    camarilla_low = pd.Series(df_1d['low'].values).shift(1).values
    camarilla_close = pd.Series(df_1d['close'].values).shift(1).values
    
    r3 = camarilla_close + (camarilla_high - camarilla_low) * 1.1 / 2
    s3 = camarilla_close - (camarilla_high - camarilla_low) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe (wait for completed 1d bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume regime: current 4h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_34_1d_aligned[i]
        r3_level = r3_aligned[i]
        s3_level = s3_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_uptrend = close_val > ema_trend
        is_downtrend = close_val < ema_trend
        
        # Entry logic
        if position == 0:
            # Long: break above R3 with volume spike in uptrend
            if close_val > r3_level and vol_spike and is_uptrend:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: break below S3 with volume spike in downtrend
            elif close_val < s3_level and vol_spike and is_downtrend:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long exit: price breaks below S3 OR trend turns down
            if close_val < s3_level or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R3 OR trend turns up
            if close_val > r3_level or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals