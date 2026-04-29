#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Long when price breaks above Camarilla R3 AND price > 4h EMA50 AND volume > 1.8x 20-period average
# Short when price breaks below Camarilla S3 AND price < 4h EMA50 AND volume > 1.8x 20-period average
# Uses ATR-based trailing stop (2.0x ATR) for risk management
# Discrete position sizing (0.20) to minimize fee churn
# Session filter: 08-20 UTC to reduce noise trades
# Target: 15-37 trades/year on 1h timeframe to avoid fee drag while capturing strong breakouts
# Works in bull markets via long Camarilla breakouts with 4h uptrend
# Works in bear markets via short Camarilla breakdowns with 4h downtrend
# Volume confirmation ensures breakouts have strong participation

name = "1h_Camarilla_R3_S3_Breakout_4hEMA50_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Session filter: 08-20 UTC (precompute hours array)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
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
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    start_idx = 50  # warmup for EMA and ATR
    
    for i in range(start_idx, n):
        # Skip if outside session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_4h = ema_50_4h_aligned[i]
        curr_atr = atr[i]
        
        # Calculate Camarilla pivot points (using previous day's OHLC)
        # Need to get daily data for pivot calculation
        if i >= 24:  # Assuming ~24 1h bars in a day
            # Get previous day's OHLC (24 bars ago)
            prev_day_high = np.max(high[i-24:i])
            prev_day_low = np.min(low[i-24:i])
            prev_day_close = close[i-1]  # Previous bar's close as proxy for day's close
            # Simplified: use recent 24-bar high/low and current close
            # In practice, would need actual daily data, but we approximate with 24-period
            day_high = prev_day_high
            day_low = prev_day_low
            day_close = curr_close
        else:
            # Not enough data for pivot calculation
            day_high = curr_high
            day_low = curr_low
            day_close = curr_close
        
        # Calculate Camarilla levels
        pivot = (day_high + day_low + day_close) / 3
        range_val = day_high - day_low
        camarilla_r3 = pivot + (range_val * 1.1 / 4)
        camarilla_s3 = pivot - (range_val * 1.1 / 4)
        
        # Volume spike confirmation: current volume > 1.8x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 1.8 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            # Trailing stop: 2.0 * ATR below highest high
            stop_price = highest_high_since_entry - 2.0 * curr_atr
            # Exit conditions: price below trailing stop OR price breaks below Camarilla S3
            if curr_close < stop_price or curr_close < camarilla_s3:
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
            # Exit conditions: price above trailing stop OR price breaks above Camarilla R3
            if curr_close > stop_price or curr_close > camarilla_r3:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 AND price > 4h EMA50 AND volume spike
            if curr_close > camarilla_r3 and curr_close > curr_ema_4h and vol_spike:
                signals[i] = 0.20
                position = 1
                highest_high_since_entry = curr_high
            # Short entry: price breaks below Camarilla S3 AND price < 4h EMA50 AND volume spike
            elif curr_close < camarilla_s3 and curr_close < curr_ema_4h and vol_spike:
                signals[i] = -0.20
                position = -1
                lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals