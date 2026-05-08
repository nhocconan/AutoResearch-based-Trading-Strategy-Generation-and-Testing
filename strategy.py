#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot level touch + 1d EMA34 trend + volume confirmation
# Long when price touches or exceeds Camarilla S3 level during uptrend (price > EMA34)
# Short when price touches or exceeds Camarilla R3 level during downtrend (price < EMA34)
# Volume filter: current 1d volume > 1.5x 20-day EMA of volume
# Designed for 4h timeframe to target 20-40 trades/year (80-160 total over 4 years)
# Camarilla levels provide precise support/resistance; EMA34 filters trend direction
# Volume confirmation ensures breakouts have institutional participation

name = "4h_Camarilla_R3S3_Touch_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and volume filters
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(df_daily['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d volume EMA (20-period) for volume filter
    vol_ema_20 = pd.Series(df_daily['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align daily indicators to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_daily, ema_34)
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_daily, vol_ema_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate Camarilla levels from previous day's OHLC
    # R3 = close + 1.1*(high - low)
    # S3 = close - 1.1*(high - low)
    camarilla_r3 = df_daily['close'] + 1.1 * (df_daily['high'] - df_daily['low'])
    camarilla_s3 = df_daily['close'] - 1.1 * (df_daily['high'] - df_daily['low'])
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_s3.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for EMA34
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(ema_34_aligned[i]) or np.isnan(vol_ema_20_aligned[i]) or \
           np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current daily volume > 1.5x 20-day EMA
        # Find the most recent completed daily bar
        idx_daily = 0
        while idx_daily < len(df_daily) and df_daily.iloc[idx_daily]['open_time'] <= prices.iloc[i]['open_time']:
            idx_daily += 1
        idx_daily -= 1  # last completed daily bar
        
        if idx_daily < 0:
            vol_filter = False
        else:
            vol_daily_current = df_daily.iloc[idx_daily]['volume']
            vol_filter = vol_daily_current > 1.5 * vol_ema_20_aligned[i]
        
        if position == 0:
            # Look for entry: price touches/exceeds Camarilla S3/R3 + trend + volume
            long_condition = (low[i] <= camarilla_s3_aligned[i] or close[i] <= camarilla_s3_aligned[i]) and \
                           close[i] > ema_34_aligned[i] and vol_filter
            short_condition = (high[i] >= camarilla_r3_aligned[i] or close[i] >= camarilla_r3_aligned[i]) and \
                            close[i] < ema_34_aligned[i] and vol_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below EMA34 or touches R3 (reversal signal)
            if close[i] < ema_34_aligned[i] or high[i] >= camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above EMA34 or touches S3 (reversal signal)
            if close[i] > ema_34_aligned[i] or low[i] <= camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals