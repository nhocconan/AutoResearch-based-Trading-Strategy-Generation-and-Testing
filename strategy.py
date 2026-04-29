#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Donchian breakout captures momentum in both bull and bear markets
# 1d EMA50 ensures we trade with higher timeframe trend
# Volume confirmation (>1.5x 20-period average) ensures institutional participation
# ATR-based stoploss (2.0x ATR) manages risk in volatile markets
# Designed for ~25-40 trades/year on 4h timeframe to minimize fee drag

name = "4h_Donchian20_1dEMA50_VolumeConfirm_ATRStop_v1"
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
    
    # Get 1d data for EMA50 trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR (14-period) for stoploss
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop = 0.0
    
    start_idx = 20  # Donchian and ATR warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_atr = atr[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR stoploss hit
            if curr_low <= lowest_low[i] or curr_close <= entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR stoploss hit
            if curr_high >= highest_high[i] or curr_close >= entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().iloc[i]
            vol_confirm = curr_volume > 1.5 * vol_ma_20
            
            # Long entry: price breaks above Donchian high in uptrend (price > 1d EMA50)
            if vol_confirm and curr_close > curr_ema50_1d:
                if curr_high > highest_high[i]:
                    signals[i] = 0.30
                    position = 1
                    entry_price = curr_close
                    atr_stop = entry_price - 2.0 * curr_atr
            # Short entry: price breaks below Donchian low in downtrend (price < 1d EMA50)
            elif vol_confirm and curr_close < curr_ema50_1d:
                if curr_low < lowest_low[i]:
                    signals[i] = -0.30
                    position = -1
                    entry_price = curr_close
                    atr_stop = entry_price + 2.0 * curr_atr
            else:
                signals[i] = 0.0
    
    return signals