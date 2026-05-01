#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Uses 1d EMA50 for stronger trend alignment to reduce whipsaws in bear markets.
# Volume > 1.8x 20-period average confirms strong momentum breakout.
# ATR-based stoploss (2.5x) manages risk.
# Target: 20-40 trades/year by tightening entry conditions (volume 1.8x, 1d trend filter).

name = "4h_Donchian20_1dEMA50_VolumeConfirm_ATRStop_v1"
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
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d data
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for 4h timeframe stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 100  # warmup for EMA and ATR
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema = ema_50_aligned[i]
        curr_atr = atr[i]
        
        # Volume confirmation: volume > 1.8x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > (1.8 * vol_ma_20)
        else:
            volume_confirm = False
        
        # Calculate Donchian channels (20-period) for 4h timeframe
        if i >= 20:
            highest_high = np.max(high[i-20:i])
            lowest_low = np.min(low[i-20:i])
            donchian_upper = highest_high
            donchian_lower = lowest_low
        else:
            donchian_upper = curr_close
            donchian_lower = curr_close
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper, price above 1d EMA50, volume spike
            if (curr_close > donchian_upper and 
                curr_close > curr_ema and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below Donchian lower, price below 1d EMA50, volume spike
            elif (curr_close < donchian_lower and 
                  curr_close < curr_ema and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions: price breaks below Donchian lower OR stoploss hit
            if (curr_close < donchian_lower or 
                curr_close < entry_price - 2.5 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: price breaks above Donchian upper OR stoploss hit
            if (curr_close > donchian_upper or 
                curr_close > entry_price + 2.5 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals