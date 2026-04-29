#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Long when price breaks above Camarilla R3 AND price > 4h EMA50 AND volume > 2.0x 24-period average
# Short when price breaks below Camarilla S3 AND price < 4h EMA50 AND volume > 2.0x 24-period average
# Uses ATR-based trailing stop (2.0x ATR) for risk management
# Discrete position sizing (0.20) to minimize fee drag
# Session filter: 08-20 UTC to avoid low-liquidity hours
# Target: 15-35 trades/year on 1h timeframe (~60-140 total over 4 years)
# Works in bull markets via long breakouts with 4h uptrend
# Works in bear markets via short breakdowns with 4h downtrend

name = "1h_Camarilla_R3_S3_Breakout_4hEMA50_VolumeConfirm_v1"
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
    
    # Pre-compute session filter (UTC hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Camarilla levels from previous 4h bar (using 4h data)
    # Camarilla: R3 = C + ((H-L)*1.1/4), S3 = C - ((H-L)*1.1/4)
    # We use previous 4h bar's OHLC to calculate current levels
    prev_close_4h = df_4h['close'].shift(1).values
    prev_high_4h = df_4h['high'].shift(1).values
    prev_low_4h = df_4h['low'].shift(1).values
    
    # Calculate Camarilla R3 and S3 levels
    camarilla_range_4h = prev_high_4h - prev_low_4h
    camarilla_R3_4h = prev_close_4h + (camarilla_range_4h * 1.1 / 4)
    camarilla_S3_4h = prev_close_4h - (camarilla_range_4h * 1.1 / 4)
    
    # Align Camarilla levels to 1h timeframe
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_R3_4h)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_S3_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    start_idx = max(100, 50, 50)  # warmup for EMA and ATR
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_4h = ema_50_4h_aligned[i]
        curr_atr = atr[i]
        curr_R3 = camarilla_R3_aligned[i]
        curr_S3 = camarilla_S3_aligned[i]
        
        # Skip if Camarilla levels are not available
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
                signals[i] = 0.20
                
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
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 AND price > 4h EMA50 AND volume spike
            if curr_close > curr_R3 and curr_close > curr_ema_4h and vol_spike:
                signals[i] = 0.20
                position = 1
                highest_high_since_entry = curr_high
            # Short entry: price breaks below Camarilla S3 AND price < 4h EMA50 AND volume spike
            elif curr_close < curr_S3 and curr_close < curr_ema_4h and vol_spike:
                signals[i] = -0.20
                position = -1
                lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals