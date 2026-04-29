#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(15) breakout with 1d EMA34 trend filter and volume confirmation
# Uses moderate Donchian channels to capture sustained moves with controlled trade frequency
# 1d EMA34 provides medium-term trend filter to avoid counter-trend trades in both bull/bear markets
# Volume > 1.3x average confirms institutional participation and reduces false breakouts
# Discrete position sizing (0.25) with ATR-based stoploss (2.0 ATR) for risk management
# Designed for ~12-37 trades/year (50-150 total over 4 years) to minimize fee drag
# Works in bull/bear via trend filter - only trades in direction of 1d EMA34

name = "12h_Donchian15_1dEMA34_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 12h Donchian channels (15-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=15, min_periods=15).max().values
    donchian_lower = low_series.rolling(window=15, min_periods=15).min().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    start_idx = max(15, 34, 20, 14)  # Donchian, EMA34, volume MA, ATR warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_donchian_upper = donchian_upper[i]
        curr_donchian_lower = donchian_lower[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        curr_atr = atr[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: stoploss hit or Donchian breakout in opposite direction
            if curr_low <= entry_price - 2.0 * curr_atr or curr_high >= curr_donchian_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: stoploss hit or Donchian breakout in opposite direction
            if curr_high >= entry_price + 2.0 * curr_atr or curr_low <= curr_donchian_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.3x 20-period average
            vol_confirmed = curr_volume > 1.3 * curr_vol_ma
            
            # Long when price breaks above 15-period Donchian upper, 1d EMA34 up-trend, volume confirmed
            if curr_high > curr_donchian_upper and curr_close > curr_ema34_1d and vol_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short when price breaks below 15-period Donchian lower, 1d EMA34 down-trend, volume confirmed
            elif curr_low < curr_donchian_lower and curr_close < curr_ema34_1d and vol_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals