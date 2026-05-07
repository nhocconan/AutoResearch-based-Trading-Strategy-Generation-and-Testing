#!/usr/bin/env python3
# 1d_1w_Camarilla_R3_S3_Breakout_Trend_Volume
# Hypothesis: Weekly trend filter + daily Camarilla R3/S3 breakout with volume confirmation.
# Uses weekly EMA50 to determine trend: price above weekly EMA50 = uptrend (long bias),
# price below = downtrend (short bias). In uptrend, long on break above daily R3 with volume;
# in downtrend, short on break below daily S3 with volume. Weekly trend filter reduces
# whipsaws in ranging markets. Target: 15-25 trades/year (60-100 over 4 years) with position
# size 0.25. Works in both bull (trend following) and bear (counter-trend reversals at
# weekly extremes) markets by aligning with higher timeframe momentum.

name = "1d_1w_Camarilla_R3_S3_Breakout_Trend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    # Camarilla R3 and S3 levels
    # R3 = close + 1.1*(high-low)*1.1/4
    # S3 = close - 1.1*(high-low)*1.1/4
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 4
    
    # Volume ratio: current volume / 20-day average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need 50 periods for weekly EMA50 and sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market regime from weekly EMA50
        uptrend_regime = close[i] > ema_50_1w_aligned[i]
        downtrend_regime = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation: volume > 1.5x average
        volume_confirm = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: break above daily R3 in uptrend regime + volume
            long_entry = (close[i] > camarilla_r3[i]) and uptrend_regime and volume_confirm
            # Short: break below daily S3 in downtrend regime + volume
            short_entry = (close[i] < camarilla_s3[i]) and downtrend_regime and volume_confirm
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: close crosses below previous day's close or regime changes to downtrend
            if (close[i] < prev_close[i]) or (not uptrend_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: close crosses above previous day's close or regime changes to uptrend
            if (close[i] > prev_close[i]) or (not downtrend_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals