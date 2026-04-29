#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend + volume confirmation + ATR stoploss
# Long when price breaks above 4h Donchian upper AND price > 12h EMA50 AND volume > 1.8x 20-bar avg
# Short when price breaks below 4h Donchian lower AND price < 12h EMA50 AND volume > 1.8x 20-bar avg
# Exit when price closes below/above 4h Donchian middle band or ATR-based stoploss
# Uses discrete position sizing (0.30) to balance return and drawdown. Target: 20-50 trades/year on 4h timeframe.
# Donchian channels provide clear trend structure, 12h EMA50 filters higher-timeframe trend,
# volume confirmation ensures breakout validity. This combination has worked well on SOL historically.

name = "4h_Donchian20_12hEMA50_VolumeSpike_v1"
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h data
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR(14) for stoploss and volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (highest_high + lowest_low) / 2.0
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 50)  # Donchian and EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(atr[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema50 = ema_50_12h_aligned[i]
        curr_upper = highest_high[i]
        curr_lower = lowest_low[i]
        curr_middle = donchian_middle[i]
        curr_atr = atr[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit conditions: close below middle band OR ATR stoploss
            if curr_close < curr_middle or curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit conditions: close above middle band OR ATR stoploss
            if curr_close > curr_middle or curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian upper AND price > 12h EMA50 AND volume confirmation
            if curr_close > curr_upper and curr_close > curr_ema50 and vol_conf:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
            # Short when price breaks below Donchian lower AND price < 12h EMA50 AND volume confirmation
            elif curr_close < curr_lower and curr_close < curr_ema50 and vol_conf:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals