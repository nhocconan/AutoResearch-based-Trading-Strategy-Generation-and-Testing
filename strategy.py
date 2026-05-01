#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w trend filter and volume confirmation.
# Uses 1w EMA34 as trend filter to ensure we trade with the weekly momentum.
# Volume confirmation requires current volume > 1.5 * 20-day average volume.
# Exits on opposite Camarilla level touch (R3 for shorts, S3 for longs) to capture mean reversion in ranges.
# Discrete position sizing 0.25 balances return and drawdown. Target: 30-100 trades over 4 years.
# Works in bull (buy R3 breakout with uptrend) and bear (sell S3 breakdown with downtrend).

name = "1d_CamarillaR3S3_Breakout_1wEMA34_Trend_VolumeConfirm_v1"
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
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d Camarilla pivot levels (R3, S3) - strong breakout levels
    # Based on previous day's high, low, close
    prev_daily_high = np.roll(high, 1)
    prev_daily_low = np.roll(low, 1)
    prev_daily_close = np.roll(close, 1)
    # Set first value to NaN since no previous day
    prev_daily_high[0] = np.nan
    prev_daily_low[0] = np.nan
    prev_daily_close[0] = np.nan
    
    camarilla_r3 = prev_daily_close + (prev_daily_high - prev_daily_low) * 1.1 / 4
    camarilla_s3 = prev_daily_close - (prev_daily_high - prev_daily_low) * 1.1 / 4
    
    # Volume confirmation: current volume > 1.5 * 20-day average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(34, 20)  # 34 (for EMA34 and volume MA20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(camarilla_r3[i]) or
            np.isnan(camarilla_s3[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Trend filter: 1w EMA34 direction
        uptrend = curr_close > ema_34_1w_aligned[i]
        downtrend = curr_close < ema_34_1w_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Camarilla R3/S3 levels
        r3_level = camarilla_r3[i]
        s3_level = camarilla_s3[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Close breaks above R3 AND uptrend AND volume confirmation
            if curr_close > r3_level and uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S3 AND downtrend AND volume confirmation
            elif curr_close < s3_level and downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on close touching S3 (mean reversion)
            if curr_close <= s3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on close touching R3 (mean reversion)
            if curr_close >= r3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals