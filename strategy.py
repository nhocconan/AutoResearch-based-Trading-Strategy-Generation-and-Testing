#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above 12h Donchian upper(20) AND price > 1d EMA50 AND volume > 1.5x 24-period average
# Short when price breaks below 12h Donchian lower(20) AND price < 1d EMA50 AND volume > 1.5x 24-period average
# Uses ATR-based trailing stop (2.5x ATR) for risk management
# Discrete position sizing (0.25) to minimize fee drag
# Target: 12-25 trades/year on 12h timeframe (~50-100 total over 4 years)
# Works in bull markets via long breakouts with 1d uptrend
# Works in bear markets via short breakdowns with 1d downtrend

name = "12h_Donchian_20_1dEMA50_VolumeConfirm_v1"
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
    
    # Calculate 12h Donchian channels (20-period)
    # We need to calculate this on the 12h timeframe itself
    # Since we're already on 12h timeframe, we can use rolling window directly
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    start_idx = max(100, 50, 50)  # warmup for EMA and ATR
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        curr_atr = atr[i]
        curr_donchian_upper = highest_high[i]
        curr_donchian_lower = lowest_low[i]
        
        # Skip if Donchian levels are not available
        if np.isnan(curr_donchian_upper) or np.isnan(curr_donchian_lower):
            signals[i] = 0.0
            continue
        
        # Volume spike confirmation: current volume > 1.5x 24-period average (24*12h = 12 days)
        if i >= 24:
            vol_ma_24 = np.mean(volume[i-24:i])
        else:
            vol_ma_24 = 0.0
        vol_spike = volume[i] > 1.5 * vol_ma_24 if vol_ma_24 > 0 else False
        
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
            # Long entry: price breaks above 12h Donchian upper AND price > 1d EMA50 AND volume spike
            if curr_close > curr_donchian_upper and curr_close > curr_ema_1d and vol_spike:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = curr_high
            # Short entry: price breaks below 12h Donchian lower AND price < 1d EMA50 AND volume spike
            elif curr_close < curr_donchian_lower and curr_close < curr_ema_1d and vol_spike:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals