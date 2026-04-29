#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and ATR-based position sizing
# Donchian channel breakouts capture strong momentum moves; 1d EMA50 ensures alignment with higher timeframe trend
# ATR-based stoploss (2.0x) manages risk; position size scales with volatility (inverse ATR) to normalize risk per trade
# Volume confirmation (>1.5x 20-period average) reduces false breakouts
# Designed for ~20-50 trades/year on 4h to minimize fee drag while maintaining edge in both bull and bear markets
# Works in bull via long breakouts in uptrend; works in bear via short breakouts in downtrend

name = "4h_Donchian20_1dEMA50_ATR_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR (14-period) for stoploss and position sizing
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Donchian and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_atr = atr[i]
        curr_vol_ma = vol_ma_20[i]
        curr_highest_20 = highest_20[i]
        curr_lowest_20 = lowest_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: stoploss hit (2.0x ATR) or price retests Donchian low (failed breakout)
            if curr_close < entry_price - 2.0 * curr_atr or curr_close < curr_lowest_20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit: stoploss hit (2.0x ATR) or price retests Donchian high (failed breakout)
            if curr_close > entry_price + 2.0 * curr_atr or curr_close > curr_highest_20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = position_size
                
        else:  # Flat - look for new breakout entries
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirm = curr_volume > 1.5 * curr_vol_ma
            
            # Long breakout when price closes above Donchian high with 1d EMA50 uptrend and volume confirmation
            if curr_close > curr_highest_20 and curr_close > curr_ema50_1d and vol_confirm:
                signals[i] = position_size
                position = 1
                entry_price = curr_close
            # Short breakout when price closes below Donchian low with 1d EMA50 downtrend and volume confirmation
            elif curr_close < curr_lowest_20 and curr_close < curr_ema50_1d and vol_confirm:
                signals[i] = -position_size
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals

# Calculate dynamic position size based on ATR (inverse volatility) to normalize risk
    # Base position size of 0.25 scaled by inverse ATR (normalized to 20-period median ATR)
    # This ensures smaller positions in high volatility, larger in low volatility
    if i >= 20:  # Only calculate after warmup
        atr_median = np.nanmedian(atr[20:i+1]) if np.any(~np.isnan(atr[20:i+1])) else 0.01
        if atr_median > 0:
            volatility_scalar = 0.01 / atr_median  # Normalize to ~1% ATR reference
            volatility_scalar = np.clip(volatility_scalar, 0.5, 2.0)  # Bound between 0.5x and 2x
            position_size = 0.25 * volatility_scalar
            position_size = min(position_size, 0.35)  # Cap at 0.35 max
        else:
            position_size = 0.25
    else:
        position_size = 0.25

# Note: The position_size calculation above is conceptually correct but placed incorrectly in the loop.
# Let me rewrite the function properly: