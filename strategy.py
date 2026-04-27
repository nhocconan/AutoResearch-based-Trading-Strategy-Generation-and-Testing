#!/usr/bin/env python3
"""
6h_OrderBlock_LiquidityGrab_1dTrend_v1
Hypothesis: Institutional order blocks combined with liquidity grab reversals, filtered by 1d trend, capture high-probability reversals in both bull and bear markets. Uses order blocks as institutional supply/demand zones and liquidity sweeps as smart money traps. Target: 80-120 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Detect order blocks (OB) - institutional supply/demand zones
    # Bullish OB: last down candle before up move (close < open)
    # Bearish OB: last up candle before down move (close > open)
    bullish_ob = (close < np.roll(open := prices['open'].values, 1)) & (np.roll(close, 1) > np.roll(open, 1))
    bearish_ob = (close > np.roll(open, 1)) & (np.roll(close, 1) < np.roll(open, 1))
    
    # Store OB levels (high/low of the candle)
    bullish_ob_high = np.where(bullish_ob, high, np.nan)
    bullish_ob_low = np.where(bullish_ob, low, np.nan)
    bearish_ob_high = np.where(bearish_ob, high, np.nan)
    bearish_ob_low = np.where(bearish_ob, low, np.nan)
    
    # Forward fill OB levels until invalidated
    def ffill_nan(arr):
        df = pd.Series(arr)
        return df.ffill().bfill().values
    
    bullish_ob_high_ff = ffill_nan(bullish_ob_high)
    bullish_ob_low_ff = ffill_nan(bullish_ob_low)
    bearish_ob_high_ff = ffill_nan(bearish_ob_high)
    bearish_ob_low_ff = ffill_nan(bearish_ob_low)
    
    # Detect liquidity grab (stop hunts) - price pierces OB level then reverses
    # Bullish liquidity grab: price breaks below bullish OB low then closes back above
    bullish_grab = (low < bullish_ob_low_ff) & (close > bullish_ob_low_ff)
    # Bearish liquidity grab: price breaks above bearish OB high then closes back below
    bearish_grab = (high > bearish_ob_high_ff) & (close < bearish_ob_high_ff)
    
    # Invalidate OB after grab (smart money has entered)
    bullish_ob_high_ff = np.where(bullish_grab, np.nan, bullish_ob_high_ff)
    bullish_ob_low_ff = np.where(bullish_grab, np.nan, bullish_ob_low_ff)
    bearish_ob_high_ff = np.where(bearish_grab, np.nan, bearish_ob_high_ff)
    bearish_ob_low_ff = np.where(bearish_grab, np.nan, bearish_ob_low_ff)
    
    # Re-FF after invalidation
    bullish_ob_high_ff = ffill_nan(bullish_ob_high_ff)
    bullish_ob_low_ff = ffill_nan(bullish_ob_low_ff)
    bearish_ob_high_ff = ffill_nan(bearish_ob_high_ff)
    bearish_ob_low_ff = ffill_nan(bearish_ob_low_ff)
    
    # Volume confirmation: above average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # Trend alignment: price vs 1d EMA50
    uptrend = close > ema50_1d
    downtrend = close < ema50_1d
    
    # Align all to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    bullish_ob_high_aligned = align_htf_to_ltf(prices, df_1d, bullish_ob_high_ff)
    bullish_ob_low_aligned = align_htf_to_ltf(prices, df_1d, bullish_ob_low_ff)
    bearish_ob_high_aligned = align_htf_to_ltf(prices, df_1d, bearish_ob_high_ff)
    bearish_ob_low_aligned = align_htf_to_ltf(prices, df_1d, bearish_ob_low_ff)
    bullish_grab_aligned = align_htf_to_ltf(prices, df_1d, bullish_grab)
    bearish_grab_aligned = align_htf_to_ltf(prices, df_1d, bearish_grab)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)
    uptrend_aligned = align_htf_to_ltf(prices, df_1d, uptrend)
    downtrend_aligned = align_htf_to_ltf(prices, df_1d, downtrend)
    
    signals = np.zeros(n)
    position = 0
    entry_price = 0.0
    size = 0.25
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(bullish_ob_high_aligned[i]) or 
            np.isnan(bullish_ob_low_aligned[i]) or np.isnan(bearish_ob_high_aligned[i]) or 
            np.isnan(bearish_ob_low_aligned[i]) or np.isnan(bullish_grab_aligned[i]) or 
            np.isnan(bearish_grab_aligned[i]) or np.isnan(volume_confirm_aligned[i]) or
            np.isnan(uptrend_aligned[i]) or np.isnan(downtrend_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        vol_conf = volume_confirm_aligned[i]
        uptrend = uptrend_aligned[i]
        downtrend = downtrend_aligned[i]
        bullish_grab = bullish_grab_aligned[i]
        bearish_grab = bearish_grab_aligned[i]
        bullish_ob_low = bullish_ob_low_aligned[i]
        bearish_ob_high = bearish_ob_high_aligned[i]
        
        if position == 0:
            # Long: bullish liquidity grab in uptrend
            if uptrend and vol_conf and bullish_grab and not np.isnan(bullish_ob_low):
                if close_val > bullish_ob_low:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            # Short: bearish liquidity grab in downtrend
            elif downtrend and vol_conf and bearish_grab and not np.isnan(bearish_ob_high):
                if close_val < bearish_ob_high:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Exit: price reaches opposite OB or grab fails
            if not np.isnan(bearish_ob_high) and close_val >= bearish_ob_high:
                signals[i] = 0.0
                position = 0
            elif not np.isnan(bullish_ob_low) and close_val < bullish_ob_low * 0.995:  # failed grab
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: price reaches opposite OB or grab fails
            if not np.isnan(bullish_ob_high) and close_val <= bullish_ob_high:
                signals[i] = 0.0
                position = 0
            elif not np.isnan(bearish_ob_low) and close_val > bearish_ob_low * 1.005:  # failed grab
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_OrderBlock_LiquidityGrab_1dTrend_v1"
timeframe = "6h"
leverage = 1.0