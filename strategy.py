#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 12h EMA50 trend filter and volume confirmation (>1.5x 20-period average)
# Donchian breakouts capture strong momentum moves; 12h EMA50 ensures alignment with intermediate trend
# Volume confirmation filters weak breakouts; discrete sizing (0.25) minimizes fee churn
# ATR-based stoploss (2.5x ATR) manages risk without look-ahead
# Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe

name = "4h_Donchian20_12hEMA50_VolumeConfirm_ATRStop"
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
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR(20) for stoploss and volume filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 20)  # 12h EMA50, Donchian/ATR/volume warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_12h = ema_50_12h_aligned[i]
        curr_atr = atr[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = curr_volume > 1.5 * curr_vol_ma
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Stoploss: price drops below entry - 2.5 * ATR
            # Take profit/exit: price closes below 12h EMA50 OR Donchian lower band
            if (curr_low <= entry_price - 2.5 * curr_atr or 
                curr_close < curr_ema_12h or 
                curr_close < lowest_low[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Stoploss: price rises above entry + 2.5 * ATR
            # Take profit/exit: price closes above 12h EMA50 OR Donchian upper band
            if (curr_high >= entry_price + 2.5 * curr_atr or 
                curr_close > curr_ema_12h or 
                curr_close > highest_high[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper band + above 12h EMA50 + volume confirmation
            if (curr_high > highest_high[i] and 
                curr_close > curr_ema_12h and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: price breaks below Donchian lower band + below 12h EMA50 + volume confirmation
            elif (curr_low < lowest_low[i] and 
                  curr_close < curr_ema_12h and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals