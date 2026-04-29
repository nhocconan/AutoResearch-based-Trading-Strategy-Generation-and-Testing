#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA34 trend filter + volume confirmation (>1.3x 20-period average) + ATR(14) stoploss
# Donchian breakout captures momentum; EMA34 filters trend direction; volume confirms participation; ATR stop manages risk
# Works in bull/bear: breakouts occur in both regimes, trend filter avoids counter-trend whipsaws
# Target: 100-200 total trades over 4 years (25-50/year) on 4h timeframe

name = "4h_Donchian20_1dEMA34_VolumeConfirm_ATRStop"
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
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian Channel (20) on 4h timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14) for dynamic stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: >1.3x 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 20, 20, 14)  # 1d EMA34, Donchian, volume MA, ATR warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_upper = highest_high[i]
        curr_lower = lowest_low[i]
        curr_atr = atr[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume confirmation
        vol_confirm = curr_volume > 1.3 * curr_vol_ma
        
        # Handle exits and stops
        if position == 1:  # Long position
            # Stoploss: 2*ATR below entry
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price closes below Donchian lower OR trend filter fails
            elif curr_close < curr_lower or curr_close < curr_ema_1d:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Stoploss: 2*ATR above entry
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price closes above Donchian upper OR trend filter fails
            elif curr_close > curr_upper or curr_close > curr_ema_1d:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: breakout above Donchian upper + above 1d EMA34 + volume confirmation
            if (curr_close > curr_upper and 
                curr_close > curr_ema_1d and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: breakout below Donchian lower + below 1d EMA34 + volume confirmation
            elif (curr_close < curr_lower and 
                  curr_close < curr_ema_1d and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals