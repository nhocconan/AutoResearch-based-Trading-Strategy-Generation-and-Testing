#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above R3 in bull trend (close > 1d EMA34) with volume > 2.0x 20-period MA.
# Short when price breaks below S3 in bear trend (close < 1d EMA34) with volume spike.
# Uses discrete position sizing (0.25). Camarilla pivots from 1d provide institutional levels.
# 1d trend filter reduces whipsaw vs shorter MAs. Volume confirmation ensures participation.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_Camarilla_R3_S3_Breakout_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    
    # Calculate 1d Camarilla pivots (based on previous day's OHLC)
    # Camarilla levels: R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low), etc.
    # We use R3/S3 for entries
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    camarilla_r3 = prev_close + 1.125 * (prev_high - prev_low)
    camarilla_s3 = prev_close - 1.125 * (prev_high - prev_low)
    
    # Align EMA and Camarilla levels to 6h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume regime: current 6h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_34_1d_aligned[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Camarilla breakout conditions (using current bar's levels)
        breakout_up = close_val > r3_level
        breakout_down = close_val < s3_level
        
        # Entry logic
        if position == 0:
            if is_bull_trend and breakout_up and vol_spike:
                signals[i] = 0.25
                position = 1
            elif is_bear_trend and breakout_down and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 OR trend reversal
            if close_val < s3_level or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R3 OR trend reversal
            if close_val > r3_level or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals