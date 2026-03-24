#!/usr/bin/env python3
"""
SMC Pro BTC - ICT Order Blocks & FVG [DOE]
Converted from TradingView Pine Script to Python
Single-timeframe simplified version (multi-TF requires pre-resampled data)
"""

import numpy as np
import pandas as pd

name = "SMC Pro BTC - ICT Order Blocks & FVG"
timeframe = "4h"
leverage = 1

def _detect_pivots(high, low, swing_len):
    """Detect swing highs and lows using pivot logic."""
    n = len(high)
    swing_high = np.full(n, np.nan)
    swing_low = np.full(n, np.nan)
    
    for i in range(swing_len, n - swing_len):
        window_high = high[i-swing_len:i+swing_len+1]
        window_low = low[i-swing_len:i+swing_len+1]
        
        if high[i] == np.max(window_high) and high[i] > np.max(np.concatenate([window_high[:swing_len], window_high[swing_len+1:]])):
            swing_high[i] = high[i]
        
        if low[i] == np.min(window_low) and low[i] < np.min(np.concatenate([window_low[:swing_len], window_low[swing_len+1:]])):
            swing_low[i] = low[i]
    
    return swing_high, swing_low

def generate_signals(prices):
    """
    Generate trading signals based on SMC concepts.
    
    Args:
        prices: pandas DataFrame with columns: open_time, open, high, low, close, volume
    
    Returns:
        numpy array with signals: 1=long, -1=short, 0=flat (length = len(prices))
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.int8)
    
    if n < 50:
        return signals
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_price = prices['open'].values
    
    swing_len = 10
    ob_lookback = 15
    sweep_window = 20
    rr_ratio = 2.0
    sl_buffer = 0.3
    pd_threshold = 0.80
    
    swing_high, swing_low = _detect_pivots(high, low, swing_len)
    
    last_sh = np.nan
    last_sl = np.nan
    bull_ob_top = np.nan
    bull_ob_bot = np.nan
    bull_ob_valid = False
    bear_ob_top = np.nan
    bear_ob_bot = np.nan
    bear_ob_valid = False
    bull_fvg_top = np.nan
    bull_fvg_bot = np.nan
    bull_fvg_valid = False
    bear_fvg_top = np.nan
    bear_fvg_bot = np.nan
    bear_fvg_valid = False
    bars_since_bull_sweep = 999
    bars_since_bear_sweep = 999
    active_sl = np.nan
    active_tp = np.nan
    position = 0
    
    for i in range(swing_len * 2, n):
        if not np.isnan(swing_high[i]):
            last_sh = swing_high[i]
        if not np.isnan(swing_low[i]):
            last_sl = swing_low[i]
        
        bull_bos = False
        bear_bos = False
        if not np.isnan(last_sh) and close[i] > last_sh:
            bull_bos = True
        if not np.isnan(last_sl) and close[i] < last_sl:
            bear_bos = True
        
        bull_sweep = False
        bear_sweep = False
        if not np.isnan(last_sl) and low[i] < last_sl and close[i] > last_sl:
            bull_sweep = True
            bars_since_bull_sweep = 0
        else:
            bars_since_bull_sweep += 1
        
        if not np.isnan(last_sh) and high[i] > last_sh and close[i] < last_sh:
            bear_sweep = True
            bars_since_bear_sweep = 0
        else:
            bars_since_bear_sweep += 1
        
        recent_bull_sweep = bars_since_bull_sweep <= sweep_window
        recent_bear_sweep = bars_since_bear_sweep <= sweep_window
        
        if bull_bos:
            for j in range(1, min(ob_lookback + 1, i)):
                if close[i-j] < open_price[i-j]:
                    bull_ob_top = high[i-j]
                    bull_ob_bot = low[i-j]
                    bull_ob_valid = True
                    break
        
        if bear_bos:
            for j in range(1, min(ob_lookback + 1, i)):
                if close[i-j] > open_price[i-j]:
                    bear_ob_top = high[i-j]
                    bear_ob_bot = low[i-j]
                    bear_ob_valid = True
                    break
        
        if bull_ob_valid and close[i] < bull_ob_bot:
            bull_ob_valid = False
        if bear_ob_valid and close[i] > bear_ob_top:
            bear_ob_valid = False
        
        bull_fvg_cond = False
        bear_fvg_cond = False
        if i >= 2:
            if low[i] > high[i-2] and close[i-1] > open_price[i-1]:
                bull_fvg_cond = True
            if high[i] < low[i-2] and close[i-1] < open_price[i-1]:
                bear_fvg_cond = True
        
        if bull_fvg_cond:
            bull_fvg_top = low[i]
            bull_fvg_bot = high[i-2]
            bull_fvg_valid = True
        
        if bear_fvg_cond:
            bear_fvg_top = low[i-2]
            bear_fvg_bot = high[i]
            bear_fvg_valid = True
        
        if bull_fvg_valid and close[i] < bull_fvg_bot:
            bull_fvg_valid = False
        if bear_fvg_valid and close[i] > bear_fvg_top:
            bear_fvg_valid = False
        
        in_bull_ob = bull_ob_valid and low[i] <= bull_ob_top and close[i] >= bull_ob_bot
        in_bear_ob = bear_ob_valid and high[i] >= bear_ob_bot and close[i] <= bear_ob_top
        in_bull_fvg = bull_fvg_valid and low[i] <= bull_fvg_top and close[i] >= bull_fvg_bot
        in_bear_fvg = bear_fvg_valid and high[i] >= bear_fvg_bot and close[i] <= bear_fvg_top
        
        in_bull_zone = in_bull_ob or in_bull_fvg
        in_bear_zone = in_bear_ob or in_bear_fvg
        
        in_discount = True
        in_premium = True
        if not np.isnan(last_sh) and not np.isnan(last_sl):
            swing_range = last_sh - last_sl
            if swing_range > 0:
                discount_ceil = last_sl + swing_range * pd_threshold
                premium_floor = last_sh - swing_range * pd_threshold
                in_discount = close[i] < discount_ceil
                in_premium = close[i] > premium_floor
        
        bull_candle = close[i] > open_price[i]
        bear_candle = close[i] < open_price[i]
        
        long_signal = recent_bull_sweep and in_bull_zone and bull_candle and in_discount and position == 0
        short_signal = recent_bear_sweep and in_bear_zone and bear_candle and in_premium and position == 0
        
        if long_signal:
            sl_base = bull_ob_bot if in_bull_ob else bull_fvg_bot
            if not np.isnan(sl_base):
                active_sl = sl_base * (1 - sl_buffer / 100)
                dist = close[i] - active_sl
                active_tp = close[i] + dist * rr_ratio
                position = 1
                signals[i] = 1
        
        if short_signal:
            sl_base = bear_ob_top if in_bear_ob else bear_fvg_top
            if not np.isnan(sl_base):
                active_sl = sl_base * (1 + sl_buffer / 100)
                dist = active_sl - close[i]
                active_tp = close[i] - dist * rr_ratio
                position = -1
                signals[i] = -1
        
        if position == 1 and not np.isnan(active_sl) and not np.isnan(active_tp):
            if low[i] <= active_sl or high[i] >= active_tp:
                signals[i] = 0
                position = 0
                active_sl = np.nan
                active_tp = np.nan
        
        if position == -1 and not np.isnan(active_sl) and not np.isnan(active_tp):
            if high[i] >= active_sl or low[i] <= active_tp:
                signals[i] = 0
                position = 0
                active_sl = np.nan
                active_tp = np.nan
    
    return signals

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        df = pd.read_csv(sys.argv[1])
        sigs = generate_signals(df)
        print(f"Generated {len(sigs)} signals")
        print(f"Long: {np.sum(sigs == 1)}, Short: {np.sum(sigs == -1)}, Flat: {np.sum(sigs == 0)}")
