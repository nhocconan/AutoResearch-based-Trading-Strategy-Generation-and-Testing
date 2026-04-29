#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + 1w EMA trend filter + volume confirmation
# Long when price > Alligator Jaw (13-period SMA) AND 1w EMA50 rising AND volume > 1.5x 20-day average
# Short when price < Alligator Jaw AND 1w EMA50 falling AND volume > 1.5x 20-day average
# Uses ATR-based trailing stop (2.5x ATR) for risk management
# Williams Alligator consists of three smoothed SMAs: Jaw (13), Teeth (8), Lips (5)
# Only Jaw is used for signal generation to reduce whipsaw
# 1w EMA50 provides strong trend filter to avoid counter-trend trades
# Volume confirmation ensures breakouts have institutional participation
# Designed for low trade frequency (target: 15-25 trades/year) to minimize fee drag
# Works in bull markets via long positions with 1w uptrend
# Works in bear markets via short positions with 1w downtrend

name = "1d_Williams_Alligator_1wEMA50_VolumeConfirm_v1"
timeframe = "1d"
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
    
    # Calculate Williams Alligator components
    # Alligator Jaw: 13-period SMMA (smoothed moving average) of median price
    # Alligator Teeth: 8-period SMMA of median price
    # Alligator Lips: 5-period SMMA of median price
    # We use median price = (high + low) / 2
    median_price = (high + low) / 2
    
    # Calculate SMMA (Smoothed Moving Average) - similar to Wilder's smoothing
    # SMMA today = (SMMA yesterday * (period-1) + price today) / period
    def smma(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values are smoothed
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    start_idx = max(50, 14)  # warmup for EMA and ATR
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_jaw = jaw[i]
        curr_ema_1w = ema_50_1w_aligned[i]
        curr_atr = atr[i]
        
        # Skip if Alligator Jaw is not available
        if np.isnan(curr_jaw) or np.isnan(curr_ema_1w):
            signals[i] = 0.0
            continue
        
        # Volume spike confirmation: current volume > 1.5x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 1.5 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Calculate EMA slope for trend direction (rising/falling)
        if i >= 1:
            ema_prev = ema_50_1w_aligned[i-1]
            ema_rising = curr_ema_1w > ema_prev
            ema_falling = curr_ema_1w < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            # Trailing stop: 2.5 * ATR below highest high
            stop_price = highest_high_since_entry - 2.5 * curr_atr
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
            # Trailing stop: 2.5 * ATR above lowest low
            stop_price = lowest_low_since_entry + 2.5 * curr_atr
            # Exit conditions: price above trailing stop
            if curr_close > stop_price:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price > Alligator Jaw AND 1w EMA50 rising AND volume spike
            if curr_close > curr_jaw and ema_rising and vol_spike:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = curr_high
            # Short entry: price < Alligator Jaw AND 1w EMA50 falling AND volume spike
            elif curr_close < curr_jaw and ema_falling and vol_spike:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals