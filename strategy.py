#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation (>1.5x average)
# Uses 4h timeframe with proven Donchian breakout structure
# 1d EMA50 provides robust trend filter for bull/bear markets
# Volume confirmation >1.5x 20-period average reduces false breakouts
# ATR-based trailing stoploss to limit drawdown
# Discrete position sizing: 0.30 for entries to balance return and risk
# Target: 75-200 total trades over 4 years (19-50/year)

name = "4h_Donchian20_1dEMA50_Volume_ATR_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period) from previous bar to avoid look-ahead
    # Upper band = highest high of previous 20 bars
    # Lower band = lowest low of previous 20 bars
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Breakout conditions
    breakout_up = close > donchian_upper
    breakout_down = close < donchian_lower
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    start_idx = max(100, 20, 50, 14)  # warmup for Donchian (20), EMA (50), ATR (14)
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_breakout_up = breakout_up[i]
        curr_breakout_down = breakout_down[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_atr = atr[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on breakout with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish breakout: price above Donchian upper + above 1d EMA50
                if curr_breakout_up and curr_close > curr_ema_50_1d:
                    signals[i] = 0.30
                    position = 1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
                # Bearish breakout: price below Donchian lower + below 1d EMA50
                elif curr_breakout_down and curr_close < curr_ema_50_1d:
                    signals[i] = -0.30
                    position = -1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
        
        elif position == 1:  # Long position
            # Trailing stoploss: exit if price drops below highest price since entry - 2.5 * ATR
            # Track highest price since entry
            if i == start_idx or position != 1:  # Reset tracking when position changes
                highest_since_entry = curr_close
            else:
                highest_since_entry = max(getattr(generate_signals, 'highest_since_entry', curr_close), curr_close)
            
            # Update highest price since entry
            generate_signals.highest_since_entry = highest_since_entry
            
            # Exit conditions
            if curr_close < (highest_since_entry - 2.5 * curr_atr):
                signals[i] = 0.0
                position = 0
                if hasattr(generate_signals, 'highest_since_entry'):
                    delattr(generate_signals, 'highest_since_entry')
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Trailing stoploss: exit if price rises above lowest price since entry + 2.5 * ATR
            # Track lowest price since entry
            if i == start_idx or position != -1:  # Reset tracking when position changes
                lowest_since_entry = curr_close
            else:
                lowest_since_entry = min(getattr(generate_signals, 'lowest_since_entry', curr_close), curr_close)
            
            # Update lowest price since entry
            generate_signals.lowest_since_entry = lowest_since_entry
            
            # Exit conditions
            if curr_close > (lowest_since_entry + 2.5 * curr_atr):
                signals[i] = 0.0
                position = 0
                if hasattr(generate_signals, 'lowest_since_entry'):
                    delattr(generate_signals, 'lowest_since_entry')
            else:
                signals[i] = -0.30
    
    return signals