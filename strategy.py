#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation
# Weekly Camarilla pivots (R3/S3) from 1w timeframe provide institutional structure levels
# Donchian(20) breakout on 6h captures intermediate-term momentum with clear entry/exit
# Volume confirmation (1.5x 20-period average) ensures institutional participation
# Designed for low trade frequency (~12-30 trades/year) to minimize fee drag on 6h timeframe
# Works in bull markets via long signals when price breaks above Donchian high with weekly bullish bias
# Works in bear markets via short signals when price breaks below Donchian low with weekly bearish bias
# Weekly pivot filter prevents counter-trend trades in strong weekly trends

name = "6h_Donchian_Breakout_1wCamarilla_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels (R3/S3)
    # Typical price = (high + low + close) / 3
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    typical_price_1w = (high_1w + low_1w + close_1w) / 3.0
    
    # Weekly Camarilla levels based on previous week's range
    prev_high_1w = np.concatenate([[high_1w[0]], high_1w[:-1]])
    prev_low_1w = np.concatenate([[low_1w[0]], low_1w[:-1]])
    prev_close_1w = np.concatenate([[close_1w[0]], close_1w[:-1]])
    
    weekly_range = prev_high_1w - prev_low_1w
    r3_1w = prev_close_1w + (weekly_range * 1.1 / 4.0)
    s3_1w = prev_close_1w - (weekly_range * 1.1 / 4.0)
    
    # Align weekly Camarilla levels to 6h timeframe (completed weekly bar only)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Calculate Donchian channels (20-period) on 6h
    # Upper channel = highest high over past 20 periods
    # Lower channel = lowest low over past 20 periods
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    start_idx = max(20, 34)  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r3_1w = r3_1w_aligned[i]
        curr_s3_1w = s3_1w_aligned[i]
        curr_donchian_high = highest_high[i]
        curr_donchian_low = lowest_low[i]
        curr_atr = atr[i]
        curr_vol_spike = vol_spike[i]
        
        # Handle exits and trailing stop
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            # Trailing stop: 2.0 * ATR below highest high
            stop_price = highest_high_since_entry - 2.0 * curr_atr
            # Exit conditions: price below trailing stop OR price breaks below Donchian low (failed breakout)
            if curr_close < stop_price or curr_close < curr_donchian_low:
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
            # Exit conditions: price above trailing stop OR price breaks above Donchian high (failed breakout)
            if curr_close > stop_price or curr_close > curr_donchian_high:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian high AND weekly bullish bias (price > weekly R3) AND volume spike
            if curr_close > curr_donchian_high and curr_close > curr_r3_1w and curr_vol_spike:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = curr_high
            # Short entry: Price breaks below Donchian low AND weekly bearish bias (price < weekly S3) AND volume spike
            elif curr_close < curr_donchian_low and curr_close < curr_s3_1w and curr_vol_spike:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals