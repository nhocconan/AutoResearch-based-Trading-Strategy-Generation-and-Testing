# hypothesis: 6h timeframe with 12h Camarilla pivot breakout and 1d volume confirmation, using 1d EMA50 for trend filter
# Camarilla levels from 12h provide tighter, more frequent levels than daily/weekly while avoiding noise of lower timeframes
# Volume confirmation ensures breakouts have conviction, trend filter avoids counter-trend trades
# Designed to work in both bull and bear markets by only trading with higher timeframe trend

name = "6h_Camarilla_R3_S3_12hBreakout_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla levels (R3, S3)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels (R3, S3) from previous 12h bar
    # Formula: Range = high - low
    # R3 = close + 1.1 * (high - low) / 2
    # S3 = close - 1.1 * (high - low) / 2
    prev_close = np.roll(df_12h['close'].values, 1)
    prev_high = np.roll(df_12h['high'].values, 1)
    prev_low = np.roll(df_12h['low'].values, 1)
    prev_close[0] = np.nan  # First value invalid
    
    camarilla_range = prev_high - prev_low
    r3 = prev_close + 1.1 * camarilla_range / 2
    s3 = prev_close - 1.1 * camarilla_range / 2
    
    # Align Camarilla levels to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_12h, r3)
    s3_6h = align_htf_to_ltf(prices, df_12h, s3)
    
    # Breakout conditions: price must close beyond the level
    breakout_up = close > r3_6h
    breakout_down = close < s3_6h
    
    # Get 1d data for EMA50 trend filter and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d average volume for volume filter (20-period)
    avg_volume_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    trend_up = close > ema_50_1d_aligned
    trend_down = close < ema_50_1d_aligned
    volume_filter = volume > (1.5 * avg_volume_1d_aligned)  # 1.5x average volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(breakout_up[i]) or np.isnan(breakout_down[i]) or
            np.isnan(trend_up[i]) or np.isnan(trend_down[i]) or
            np.isnan(volume_filter[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above R3 + 1d uptrend + volume confirmation
            if breakout_up[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S3 + 1d downtrend + volume confirmation
            elif breakout_down[i] and trend_down[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit conditions: price returns to 12h close or trend reversal
            # Get 12h close price aligned to 6t
            df_12h_close = df_12h['close'].values
            prev_12h_close = np.roll(df_12h_close, 1)
            prev_12h_close[0] = np.nan
            prev_12h_close_aligned = align_htf_to_ltf(prices, df_12h, prev_12h_close)
            
            if close[i] <= prev_12h_close_aligned[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: price returns to 12h close or trend reversal
            df_12h_close = df_12h['close'].values
            prev_12h_close = np.roll(df_12h_close, 1)
            prev_12h_close[0] = np.nan
            prev_12h_close_aligned = align_htf_to_ltf(prices, df_12h, prev_12h_close)
            
            if close[i] >= prev_12h_close_aligned[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals