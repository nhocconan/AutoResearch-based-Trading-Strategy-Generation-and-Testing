#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Donchian breakout captures momentum; 1d EMA34 ensures alignment with higher timeframe trend
# Volume confirmation (>1.8x 20-period average) filters weak breakouts
# ATR-based stoploss (2.5x ATR) manages risk
# Designed for ~25-40 trades/year on 4h timeframe to minimize fee drag
# Works in both bull (breakouts with trend) and bear (failed breaks reverse) markets

name = "4h_Donchian20_1dEMA34_VolumeConfirm_ATRStop_v1"
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
    
    # Get 1d data for EMA34 trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period) for breakout
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 20  # Donchian and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        curr_atr = atr[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price breaks below Donchian lower band or ATR stoploss
            if curr_low < lowest_low[i] or curr_close < entry_price - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper band or ATR stoploss
            if curr_high > highest_high[i] or curr_close > entry_price + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.8x 20-period average
            vol_confirm = curr_volume > 1.8 * curr_vol_ma
            
            # Long entry: price breaks above Donchian upper band in uptrend (price > 1d EMA34)
            if vol_confirm and curr_close > curr_ema34_1d:
                if curr_high > highest_high[i]:
                    signals[i] = 0.30
                    position = 1
                    entry_price = curr_close
            # Short entry: price breaks below Donchian lower band in downtrend (price < 1d EMA34)
            elif vol_confirm and curr_close < curr_ema34_1d:
                if curr_low < lowest_low[i]:
                    signals[i] = -0.30
                    position = -1
                    entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals