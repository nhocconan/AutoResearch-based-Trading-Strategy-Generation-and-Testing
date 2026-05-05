#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume confirmation
# Long when price breaks above Camarilla R3 AND 4h close > 4h EMA34 AND volume > 1.5 * 20-period avg volume
# Short when price breaks below Camarilla S3 AND 4h close < 4h EMA34 AND volume > 1.5 * 20-period avg volume
# Uses discrete sizing (0.20) to limit fee drag. Target: 15-37 trades/year per symbol.
# Camarilla provides intraday structure; 4h EMA34 filters trend; volume confirms breakout strength.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.
# Uses 4h for HTF trend to avoid counter-trend trades and 1h for Camarilla timing.

name = "1h_Camarilla_R3S3_4hEMA34_VolumeConfirm"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_ = prices['open'].values
    
    # Get 4h data ONCE before loop for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate 4h EMA34 for trend filter
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Uptrend when close > EMA34, downtrend when close < EMA34
    uptrend_4h = close_4h > ema_34_4h
    downtrend_4h = close_4h < ema_34_4h
    
    # Align 4h trend to 1h timeframe
    uptrend_4h_aligned = align_htf_to_ltf(prices, df_4h, uptrend_4h.astype(float))
    downtrend_4h_aligned = align_htf_to_ltf(prices, df_4h, downtrend_4h.astype(float))
    
    # Calculate Camarilla levels for 1h timeframe using previous 1h bar
    # Camarilla: based on previous bar's range
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # Avoid NaN on first bar
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    # Previous bar's range
    range_ = prev_high - prev_low
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = prev_close + range_ * 1.1 / 4
    camarilla_s3 = prev_close - range_ * 1.1 / 4
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup for volume MA
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(uptrend_4h_aligned[i]) or np.isnan(downtrend_4h_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > Camarilla R3 AND 4h uptrend AND volume confirmation
            if (close[i] > camarilla_r3[i] and 
                uptrend_4h_aligned[i] > 0.5 and 
                volume_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price < Camarilla S3 AND 4h downtrend AND volume confirmation
            elif (close[i] < camarilla_s3[i] and 
                  downtrend_4h_aligned[i] > 0.5 and 
                  volume_filter[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price < Camarilla S3 OR 4h trend changes to downtrend
            if (close[i] < camarilla_s3[i] or 
                downtrend_4h_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price > Camarilla R3 OR 4h trend changes to uptrend
            if (close[i] > camarilla_r3[i] or 
                uptrend_4h_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals