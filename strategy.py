#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation
# Uses Camarilla pivot levels (R3/S3) from 1d for clear breakout levels
# 1d EMA50 provides trend filter to avoid counter-trend trades
# Volume spike (1.8x 20-period average) confirms breakout validity
# ATR-based stoploss (2.0x ATR) manages risk
# Designed for low trade frequency (target: 50-150 total over 4 years) to minimize fee drag
# Works in bull markets via trend-following breaks and avoids counter-trend trades in bear markets

name = "12h_Camarilla_R3S3_Breakout_1dEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for EMA
    
    for i in range(start_idx, n):
        # Need at least 1 previous day for Camarilla calculation
        if i < 1:
            signals[i] = 0.0
            continue
            
        # Get previous 1d OHLC for Camarilla calculation
        prev_idx = i - 1
        if prev_idx < 0:
            signals[i] = 0.0
            continue
            
        # Calculate Camarilla levels from previous 1d bar
        prev_high = high[prev_idx]
        prev_low = low[prev_idx]
        prev_close = close[prev_idx]
        
        # Camarilla pivot levels
        pivot = (prev_high + prev_low + prev_close) / 3.0
        range_hl = prev_high - prev_low
        r3 = pivot + (range_hl * 1.1 / 4.0)  # R3 level
        s3 = pivot - (range_hl * 1.1 / 4.0)  # S3 level
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        curr_atr = atr[i]
        
        # Volume spike confirmation: current volume > 1.8x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 1.8 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry
            stop_price = entry_price - 2.0 * curr_atr
            # Exit conditions: price below S3 OR price below 1d EMA50 OR stoploss hit
            if curr_close < s3 or curr_close < curr_ema_1d or curr_close < stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry
            stop_price = entry_price + 2.0 * curr_atr
            # Exit conditions: price above R3 OR price above 1d EMA50 OR stoploss hit
            if curr_close > r3 or curr_close > curr_ema_1d or curr_close > stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above R3 AND price > 1d EMA50 AND volume spike
            if curr_high > r3 and curr_close > curr_ema_1d and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: price breaks below S3 AND price < 1d EMA50 AND volume spike
            elif curr_low < s3 and curr_close < curr_ema_1d and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals