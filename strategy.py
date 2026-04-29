#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation
# Long: price > upper Donchian(20) AND price > 1d EMA34 AND volume > 2.0x 20-bar avg
# Short: price < lower Donchian(20) AND price < 1d EMA34 AND volume > 2.0x 20-bar avg
# Exit: price crosses 20-bar EMA (middle Donchian) OR ATR stoploss (2.0 * ATR)
# Donchian channels provide clear structure with proven edge in crypto trends
# 1d EMA34 filters for higher timeframe trend alignment to avoid counter-trend trades
# Volume spike confirms institutional participation and reduces false breakouts
# ATR stoploss manages risk during volatile periods
# Discrete position sizing: 0.25 for long/short to balance exposure and fee drag
# Target: 100-180 total trades over 4 years (25-45/year) on 4h timeframe

name = "4h_Donchian_Breakout_1dEMA34_VolumeSpike_ATRStop_v1"
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
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    # Upper channel: highest high over 20 periods
    # Lower channel: lowest low over 20 periods
    # Middle channel: 20-period EMA of close (used for exit)
    lookback = 20
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    middle_ema = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        upper_channel[i] = np.max(high[i-lookback+1:i+1])
        lower_channel[i] = np.min(low[i-lookback+1:i+1])
    
    # Calculate 20-period EMA for middle channel and exit signal
    close_series = pd.Series(close)
    middle_ema = close_series.ewm(span=lookback, adjust=False, min_periods=lookback).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(lookback, 34)  # warmup for indicators
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_upper = upper_channel[i]
        curr_lower = lower_channel[i]
        curr_middle = middle_ema[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_atr = atr[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 2.0 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Stoploss: 2 * ATR below entry
            stop_price = entry_price - 2.0 * curr_atr
            # Exit conditions: price below middle Donchian (20 EMA) OR price below 1d EMA34 OR stoploss hit
            if curr_close < curr_middle or curr_close < curr_ema_1d or curr_close < stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry
            stop_price = entry_price + 2.0 * curr_atr
            # Exit conditions: price above middle Donchian (20 EMA) OR price above 1d EMA34 OR stoploss hit
            if curr_close > curr_middle or curr_close > curr_ema_1d or curr_close > stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price > upper Donchian AND price > 1d EMA34 AND volume spike
            if curr_close > curr_upper and curr_close > curr_ema_1d and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: price < lower Donchian AND price < 1d EMA34 AND volume spike
            elif curr_close < curr_lower and curr_close < curr_ema_1d and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals