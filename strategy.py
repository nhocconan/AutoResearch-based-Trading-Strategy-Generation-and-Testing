#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above Camarilla R3 AND price > 1w EMA50 AND volume > 2.0x 24-period average
# Short when price breaks below Camarilla S3 AND price < 1w EMA50 AND volume > 2.0x 24-period average
# Uses ATR-based trailing stop (2.0x ATR) for risk management
# Discrete position sizing (0.25) to balance return and fee drag
# Target: 12-30 trades/year on 12h timeframe to avoid fee drag while capturing strong breakouts
# Works in bull markets via long breakouts with 1w uptrend
# Works in bear markets via short breakdowns with 1w downtrend
# Volume confirmation ensures breakouts have strong institutional participation
# Uses proper MTF data loading: get_htf_data() called ONCE before loop

name = "12h_Camarilla_R3_S3_Breakout_1wEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    start_idx = 100  # warmup for EMA and ATR
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1w = ema_50_1w_aligned[i]
        curr_atr = atr[i]
        
        # Calculate Camarilla levels from previous 1d bar (need completed 1d bar)
        # We need to align the 1d Camarilla levels to the 12h timeframe
        # We'll calculate Camarilla levels for each 1d bar and then align them
        
        # Load 1d data for Camarilla calculation
        df_1d = get_htf_data(prices, '1d')
        if len(df_1d) < 2:
            signals[i] = 0.0
            continue
            
        # Calculate Camarilla levels for 1d data
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d_arr = df_1d['close'].values
        
        # Initialize arrays for Camarilla levels
        camarilla_R3 = np.full_like(close_1d_arr, np.nan)
        camarilla_S3 = np.full_like(close_1d_arr, np.nan)
        
        # Calculate Camarilla levels for each 1d bar (starting from index 1 to avoid lookback issues)
        for j in range(1, len(close_1d_arr)):
            # Camarilla levels use previous day's high, low, close
            H = high_1d[j-1]
            L = low_1d[j-1]
            C = close_1d_arr[j-1]
            range_hl = H - L
            
            camarilla_R3[j] = C + range_hl * 1.1 / 4
            camarilla_S3[j] = C - range_hl * 1.1 / 4
        
        # Align Camarilla levels to 12h timeframe
        camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
        camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
        
        # Get current Camarilla levels
        curr_R3 = camarilla_R3_aligned[i]
        curr_S3 = camarilla_S3_aligned[i]
        
        # Skip if Camarilla levels are not available (NaN)
        if np.isnan(curr_R3) or np.isnan(curr_S3):
            signals[i] = 0.0
            continue
        
        # Volume spike confirmation: current volume > 2.0x 24-period average
        if i >= 24:
            vol_ma_24 = np.mean(volume[i-24:i])
        else:
            vol_ma_24 = 0.0
        vol_spike = volume[i] > 2.0 * vol_ma_24 if vol_ma_24 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            # Trailing stop: 2.0 * ATR below highest high
            stop_price = highest_high_since_entry - 2.0 * curr_atr
            # Exit conditions: price below trailing stop
            if curr_close < stop_price:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            # Trailing stop: 2.0 * ATR above lowest low
            stop_price = lowest_low_since_entry + 2.0 * curr_atr
            # Exit conditions: price above trailing stop
            if curr_close > stop_price:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 AND price > 1w EMA50 AND volume spike
            if curr_close > curr_R3 and curr_close > curr_ema_1w and vol_spike:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = curr_high
            # Short entry: price breaks below Camarilla S3 AND price < 1w EMA50 AND volume spike
            elif curr_close < curr_S3 and curr_close < curr_ema_1w and vol_spike:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals