#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation
# Uses Donchian channel breakouts for trend capture, filtered by 1d EMA34 to avoid counter-trend trades
# Volume > 1.8x average confirms institutional participation and reduces false breakouts
# ATR-based stoploss (2.5x ATR) manages risk during adverse moves
# Discrete position sizing (0.25) with Donchian(10) exit for quick profit taking
# Designed for ~20-40 trades/year to minimize fee drag while capturing strong trends
# Works in bull/bear via trend filter - only trades in direction of 1d EMA34

name = "4h_Donchian20_1dEMA34_VolumeConfirm_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels (20-period for entry, 10-period for exit)
    donchian_20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_10_high = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donchian_10_low = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 34)  # Donchian20 and 1d EMA34 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_20_high[i]) or np.isnan(donchian_20_low[i]) or 
            np.isnan(donchian_10_high[i]) or np.isnan(donchian_10_low[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(atr[i])):
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
            # Exit: price below Donchian(10) low (profit taking) OR stoploss hit
            if curr_close < donchian_10_low[i] or curr_close < entry_price - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above Donchian(10) high (profit taking) OR stoploss hit
            if curr_close > donchian_10_high[i] or curr_close > entry_price + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume spike confirmation: current volume > 1.8x 20-period average
            vol_spike = curr_volume > 1.8 * curr_vol_ma
            
            # Long when price breaks above Donchian(20) high with 1d EMA34 uptrend and volume spike
            if curr_high > donchian_20_high[i] and curr_close > curr_ema34_1d and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short when price breaks below Donchian(20) low with 1d EMA34 downtrend and volume spike
            elif curr_low < donchian_20_low[i] and curr_close < curr_ema34_1d and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals