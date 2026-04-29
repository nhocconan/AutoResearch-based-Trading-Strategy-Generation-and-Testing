#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Donchian breakout captures momentum; 1w EMA50 ensures we trade with the primary trend
# Volume confirmation (>1.5x 20-period average) ensures institutional participation
# ATR-based stoploss (2.5x ATR) manages risk
# Designed for ~12-25 trades/year on 12h timeframe to minimize fee drag
# Works in both bull and bear markets by following the 1w trend

name = "12h_Donchian20_1wEMA50_VolumeConfirm_ATRStop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter (HTF = 1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian(20) channels on 12h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for stoploss
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period average volume for confirmation (on 12h data)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop = 0.0
    
    start_idx = 20  # Donchian and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_ema50_1w = ema_50_1w_aligned[i]
        curr_upper = highest_high[i]
        curr_lower = lowest_low[i]
        curr_atr = atr[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits based on stoploss or Donchian opposite breakout
        if position == 1:  # Long position
            # Exit: stoploss hit or price breaks below lower Donchian band
            if curr_low <= entry_price - 2.5 * curr_atr or curr_close < curr_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: stoploss hit or price breaks above upper Donchian band
            if curr_high >= entry_price + 2.5 * curr_atr or curr_close > curr_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirm = curr_volume > 1.5 * curr_vol_ma
            
            # Long entry: price breaks above upper Donchian band in uptrend (price > 1w EMA50)
            if vol_confirm and curr_close > curr_ema50_1w:
                if curr_high > curr_upper:  # Breakout confirmation
                    signals[i] = 0.30
                    position = 1
                    entry_price = curr_close
                    atr_stop = entry_price - 2.5 * curr_atr
            # Short entry: price breaks below lower Donchian band in downtrend (price < 1w EMA50)
            elif vol_confirm and curr_close < curr_ema50_1w:
                if curr_low < curr_lower:  # Breakdown confirmation
                    signals[i] = -0.30
                    position = -1
                    entry_price = curr_close
                    atr_stop = entry_price + 2.5 * curr_atr
            else:
                signals[i] = 0.0
    
    return signals