#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation
# Long: Close > Donchian Upper(20) AND price > 1d EMA34 AND volume > 2.0x 20-bar avg
# Short: Close < Donchian Lower(20) AND price < 1d EMA34 AND volume > 2.0x 20-bar avg
# Exit: Price crosses 1d EMA34 (trend change) OR ATR-based stoploss (2.0x ATR)
# Works in bull via breakout continuation, in bear via mean reversion at extremes
# Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe

name = "4h_Donchian_Breakout_1dEMA34_VolumeSpike_ATRStop_v2"
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
    
    # Load HTF data ONCE before loop
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
    
    # Calculate Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(34, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_atr = atr[i]
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Handle exits and stops
        if position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Exit: Price crosses below 1d EMA34 (trend change)
            elif curr_close < curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Exit: Price crosses above 1d EMA34 (trend change)
            elif curr_close > curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Close > Donchian Upper AND price > 1d EMA34 AND volume spike
            if (curr_close > donch_high[i] and 
                curr_close > curr_ema_1d and
                vol_spike):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: Close < Donchian Lower AND price < 1d EMA34 AND volume spike
            elif (curr_close < donch_low[i] and 
                  curr_close < curr_ema_1d and
                  vol_spike):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals