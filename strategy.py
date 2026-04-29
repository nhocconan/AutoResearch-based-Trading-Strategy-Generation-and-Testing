#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Donchian breakouts capture strong momentum moves; 1d EMA50 ensures alignment with higher timeframe trend
# Volume spike (>1.8x 20-period average) confirms institutional participation and reduces false breakouts
# ATR-based stoploss (2.0x ATR) manages risk during adverse moves
# Discrete position sizing (0.30) balances return potential with fee minimization
# Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe

name = "4h_Donchian20_1dEMA50_VolumeConfirm_v1"
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
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for stoploss and position sizing
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for spike confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 20, 14)  # 1d EMA50, Donchian(20), ATR(14) warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        curr_atr = atr[i]
        
        # Volume spike confirmation: current volume > 1.8x 20-period average
        vol_spike = curr_volume > 1.8 * curr_vol_ma
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR breaks 1d EMA50 trend
            # Stoploss: price closes below entry - 2.0 * ATR
            if (curr_close < curr_donchian_low or 
                curr_close < curr_ema_1d or 
                curr_close < entry_price - 2.0 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR breaks 1d EMA50 trend
            # Stoploss: price closes above entry + 2.0 * ATR
            if (curr_close > curr_donchian_high or 
                curr_close > curr_ema_1d or 
                curr_close > entry_price + 2.0 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high AND above 1d EMA50 AND volume spike
            if curr_high > curr_donchian_high and curr_close > curr_ema_1d and vol_spike:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
            # Short entry: price breaks below Donchian low AND below 1d EMA50 AND volume spike
            elif curr_low < curr_donchian_low and curr_close < curr_ema_1d and vol_spike:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals