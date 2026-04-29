#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when close > upper Donchian AND price > 1w EMA50 AND volume > 1.5x 20-bar avg
# Short when close < lower Donchian AND price < 1w EMA50 AND volume > 1.5x 20-bar avg
# Exit on opposite Donchian level OR ATR-based stoploss (2.0x ATR)
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 10-20 trades/year on 1d.
# Donchian channels provide clear breakout levels. 1w EMA50 filters counter-trend moves on weekly trend.
# Volume confirmation ensures institutional participation. ATR stoploss manages risk.
# This strategy avoids overtrading by requiring confluence of 3 strong conditions on daily timeframe.

name = "1d_Donchian20_1wEMA50_VolumeConfirm_ATRStop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA(50) on 1w data
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    # Upper channel = highest high over past 20 periods
    # Lower channel = lowest low over past 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_donchian = high_series.rolling(window=20, min_periods=20).max().values
    lower_donchian = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 20, 14)  # EMA50 warmup, Donchian, ATR all need warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        ema_50 = ema_50_1w_aligned[i]
        upper_donch = upper_donchian[i]
        lower_donch = lower_donchian[i]
        atr_val = atr[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Check stoploss: close < entry_price - 2.0 * ATR
            if curr_close < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            # Check exit: close < lower Donchian (opposite level)
            elif curr_close < lower_donch:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Check stoploss: close > entry_price + 2.0 * ATR
            if curr_close > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            # Check exit: close > upper Donchian (opposite level)
            elif curr_close > upper_donch:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when close > upper Donchian AND price > 1w EMA50 AND volume confirmation
            if curr_close > upper_donch and curr_close > ema_50 and vol_conf:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short when close < lower Donchian AND price < 1w EMA50 AND volume confirmation
            elif curr_close < lower_donch and curr_close < ema_50 and vol_conf:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals