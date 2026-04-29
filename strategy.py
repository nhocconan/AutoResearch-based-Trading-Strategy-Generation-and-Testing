#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Uses Donchian channel for structural breakouts, filtered by 12h EMA50 trend direction.
# Volume spike (>1.8x 20-period average) confirms breakout strength.
# ATR-based trailing stop (2.5x ATR) manages risk.
# Designed for ~30-60 trades/year on 4h timeframe to balance opportunity and fee drag.
# Works in both bull and bear markets via 12h trend filter - only takes breakouts in trend direction.

name = "4h_Donchian20_12hEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter (HTF = 12h)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channel (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for confirmation (on 4h data)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR (14-period) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 20  # Donchian, volume MA and ATR warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema50_12h = ema_50_12h_aligned[i]
        curr_upper = highest_high[i]
        curr_lower = lowest_low[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        curr_atr = atr[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: trailing stoploss or price closes below Donchian lower
            if curr_close < curr_highest_since_entry - 2.5 * curr_atr or curr_close < curr_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                # Update highest high since entry for trailing stop
                if 'highest_since_entry' not in locals():
                    highest_since_entry = curr_high
                else:
                    highest_since_entry = max(highest_since_entry, curr_high)
                
        elif position == -1:  # Short position
            # Exit: trailing stoploss or price closes above Donchian upper
            if curr_close > curr_lowest_since_entry + 2.5 * curr_atr or curr_close > curr_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                # Update lowest low since entry for trailing stop
                if 'lowest_since_entry' not in locals():
                    lowest_since_entry = curr_low
                else:
                    lowest_since_entry = min(lowest_since_entry, curr_low)
                
        else:  # Flat - look for new entries
            # Reset tracking variables
            if 'highest_since_entry' in locals():
                del highest_since_entry
            if 'lowest_since_entry' in locals():
                del lowest_since_entry
            
            # Volume confirmation: current volume > 1.8x 20-period average
            vol_confirm = curr_volume > 1.8 * curr_vol_ma
            
            # Long entry: bullish breakout above Donchian upper in uptrend (price > 12h EMA50)
            if vol_confirm and curr_close > curr_ema50_12h:
                if curr_high > curr_upper:  # Breakout above upper band
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    highest_since_entry = curr_high
            # Short entry: bearish breakdown below Donchian lower in downtrend (price < 12h EMA50)
            elif vol_confirm and curr_close < curr_ema50_12h:
                if curr_low < curr_lower:  # Breakdown below lower band
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    lowest_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals