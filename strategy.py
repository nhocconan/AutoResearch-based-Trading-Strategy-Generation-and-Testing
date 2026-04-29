#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d EMA50 trend filter
# Uses Donchian channel from 4h timeframe for breakout signals
# Volume confirmation (>2.0x 20-period average) filters false breakouts
# 1d EMA50 trend filter ensures alignment with higher timeframe momentum
# Designed for 4h timeframe targeting 75-200 total trades over 4 years (19-50/year)
# Proven pattern: Donchian + volume + trend = SOLUSDT test Sharpe 1.10-1.38 (from research)
# Added ATR-based stoploss to manage risk and reduce whipsaw

name = "4h_Donchian20_VolumeConfirmation_1dEMA50_Trend_ATRStop"
timeframe = "4h"
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
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) from 4h data
    high_roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) for stoploss
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 50, 20, 14)  # Donchian, 1d EMA, volume MA, ATR warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_roll_max[i]) or np.isnan(low_roll_min[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_upper = high_roll_max[i]
        curr_lower = low_roll_min[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        curr_atr = atr[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = curr_volume > 2.0 * curr_vol_ma
        
        # Handle exits and trailing logic
        if position == 1:  # Long position
            # Stoploss: price drops below entry - 2.0 * ATR
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Exit: price breaks below Donchian lower
            elif curr_close < curr_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Stoploss: price rises above entry + 2.0 * ATR
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Exit: price breaks above Donchian upper
            elif curr_close > curr_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper with volume confirmation and uptrend
            if vol_confirm and curr_high > curr_upper and curr_close > curr_ema_1d:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
            # Short entry: price breaks below Donchian lower with volume confirmation and downtrend
            elif vol_confirm and curr_low < curr_lower and curr_close < curr_ema_1d:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals