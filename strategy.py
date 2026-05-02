#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA50 trend and volume confirmation
# Camarilla R3/S3 levels provide strong breakout thresholds with 1w EMA50 filtering for primary trend.
# Breakouts above R3 or below S3 with 1w EMA50 trend alignment capture sustained momentum.
# Volume confirmation (1.8x 24-period average) filters false breakouts in choppy markets.
# Works in both bull/bear markets by only taking breakouts aligned with 1w EMA50.
# Discrete sizing 0.28 targets ~75-125 trades over 4 years (19-31/year) for 12h timeframe.

name = "12h_Camarilla_R3S3_Breakout_1wEMA50_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1w
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for Camarilla levels (more stable than lower timeframes)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day (using 1d OHLC)
    # Camarilla: R3 = close + 1.125*(high-low), S3 = close - 1.125*(high-low)
    prev_1d_close = df_1d['close'].shift(1).values
    prev_1d_high = df_1d['high'].shift(1).values
    prev_1d_low = df_1d['low'].shift(1).values
    camarilla_r3 = prev_1d_close + 1.125 * (prev_1d_high - prev_1d_low)
    camarilla_s3 = prev_1d_close - 1.125 * (prev_1d_high - prev_1d_low)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation (1.8x 24-period average on 12h timeframe)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for calculations)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long breakout: price > R3 with 1w uptrend (close > EMA50)
            long_breakout = close[i] > camarilla_r3_aligned[i]
            # Short breakdown: price < S3 with 1w downtrend (close < EMA50)
            short_breakout = close[i] < camarilla_s3_aligned[i]
            
            # 1w EMA50 trend filter: close above/below EMA indicates trend direction
            ema_trend_up = close[i] > ema_50_1w_aligned[i]
            ema_trend_down = close[i] < ema_50_1w_aligned[i]
            
            if long_breakout and ema_trend_up and volume_spike[i]:
                signals[i] = 0.28
                position = 1
            elif short_breakout and ema_trend_down and volume_spike[i]:
                signals[i] = -0.28
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < S3 or trend reversal (close < EMA50)
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        
        elif position == -1:  # Short position
            # Exit: price > R3 or trend reversal (close > EMA50)
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals