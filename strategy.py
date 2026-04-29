#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Uses tighter Donchian channels for fewer, higher-quality breakouts
# Volume confirmation > 1.5x average to filter weak breakouts
# 1d EMA34 trend filter ensures alignment with higher timeframe momentum
# Discrete position sizing (0.25) and ATR-based stoploss to manage risk
# Designed to reduce trade frequency while maintaining edge in both bull and bear markets

name = "4h_Donchian20_1dEMA34_VolumeSpike_v1"
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
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels (20-period)
    # Upper band = highest high of last 20 periods
    # Lower band = lowest low of last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # Track entry price for stoploss
    
    start_idx = max(20, 34, 14)  # Donchian, EMA34, and ATR warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_atr = atr[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: stoploss hit or Donchian lower band break
            if curr_low <= entry_price - 2.0 * curr_atr or curr_close < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: stoploss hit or Donchian upper band break
            if curr_high >= entry_price + 2.0 * curr_atr or curr_close > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
            vol_confirmed = curr_volume > 1.5 * vol_ma_20
            
            # Long when price breaks above Donchian upper band, 1d EMA34 up-trend, volume confirmed
            if curr_high > donchian_upper[i] and curr_close > curr_ema34_1d and vol_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = close[i+1] if i+1 < n else close[i]  # Enter at next open
            # Short when price breaks below Donchian lower band, 1d EMA34 down-trend, volume confirmed
            elif curr_low < donchian_lower[i] and curr_close < curr_ema34_1d and vol_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = close[i+1] if i+1 < n else close[i]  # Enter at next open
            else:
                signals[i] = 0.0
    
    return signals