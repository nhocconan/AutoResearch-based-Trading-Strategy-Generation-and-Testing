#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation (>1.8x average)
# Uses 1h timeframe with strict entry conditions to target 15-37 trades/year (60-150 over 4 years)
# 4h EMA50 provides trend filter to avoid counter-trend trades in both bull/bear markets
# Volume confirmation ensures breakouts have institutional participation
# Session filter (08-20 UTC) reduces noise during low-liquidity periods
# Discrete position sizing: 0.20 for entries to minimize fee impact

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 1h Camarilla levels (R3, S3) from previous bar
    hl_range = high - low
    camarilla_r3 = close + 1.1 * hl_range
    camarilla_s3 = close - 1.1 * hl_range
    
    # Need previous bar's levels to avoid look-ahead
    camarilla_r3_prev = np.roll(camarilla_r3, 1)
    camarilla_s3_prev = np.roll(camarilla_s3, 1)
    camarilla_r3_prev[0] = np.nan
    camarilla_s3_prev[0] = np.nan
    
    # Breakout conditions
    breakout_up = close > camarilla_r3_prev
    breakout_down = close < camarilla_s3_prev
    
    # Volume confirmation: volume > 1.8x 20-period average (balanced to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20, 50)  # warmup for Camarilla (20), volume MA (20), EMA (50)
    
    for i in range(start_idx, n):
        # Skip if not in trading session or indicators not ready
        if (not in_session[i] or 
            np.isnan(camarilla_r3_prev[i]) or 
            np.isnan(camarilla_s3_prev[i]) or
            np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_breakout_up = breakout_up[i]
        curr_breakout_down = breakout_down[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_50_4h = ema_50_4h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on breakout with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish breakout: price above Camarilla R3 + above 4h EMA50
                if curr_breakout_up and curr_close > curr_ema_50_4h:
                    signals[i] = 0.20
                    position = 1
                # Bearish breakout: price below Camarilla S3 + below 4h EMA50
                elif curr_breakout_down and curr_close < curr_ema_50_4h:
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: price closes below Camarilla S3 (reversal) or above Camarilla R3 (take profit)
            if curr_close < camarilla_s3_prev[i] or curr_close > camarilla_r3_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price closes above Camarilla R3 (reversal) or below Camarilla S3 (take profit)
            if curr_close > camarilla_r3_prev[i] or curr_close < camarilla_s3_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals