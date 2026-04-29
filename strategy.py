#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation (>2.0x 20-period average)
# Donchian channels identify breakouts; 1d EMA34 filters direction; volume confirmation ensures institutional participation
# ATR-based stoploss (2.5x ATR) limits drawdown; discrete sizing (0.30) minimizes fee churn
# Works in both bull/bear markets: breakouts capture volatility expansion in trending regimes
# Target: 80-180 total trades over 4 years (20-45/year) on 4h timeframe

name = "4h_Donchian_Breakout_1dEMA34_VolumeConfirm_v2"
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
    
    # Donchian Channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR (14-period) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 20-period average volume for confirmation (on 4h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    start_idx = max(50, 20, 20, 14)  # 1d EMA34, Donchian, volume MA, ATR warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_high_20 = high_20[i]
        curr_low_20 = low_20[i]
        curr_atr = atr[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = curr_volume > 2.0 * curr_vol_ma
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Stoploss: price closes below entry - 2.5 * ATR
            if curr_close < entry_price - 2.5 * atr_at_entry:
                signals[i] = 0.0
                position = 0
            # Exit: price closes below Donchian low OR Donchian breakout in opposite direction
            elif curr_close < curr_low_20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Stoploss: price closes above entry + 2.5 * ATR
            if curr_close > entry_price + 2.5 * atr_at_entry:
                signals[i] = 0.0
                position = 0
            # Exit: price closes above Donchian high OR Donchian breakout in opposite direction
            elif curr_close > curr_high_20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Long entry: breakout above Donchian high + above 1d EMA34 + volume confirmation
            if (curr_close > curr_high_20 and 
                curr_close > curr_ema_1d and 
                vol_confirm):
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
                atr_at_entry = curr_atr
            # Short entry: breakout below Donchian low + below 1d EMA34 + volume confirmation
            elif (curr_close < curr_low_20 and 
                  curr_close < curr_ema_1d and 
                  vol_confirm):
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
                atr_at_entry = curr_atr
            else:
                signals[i] = 0.0
    
    return signals