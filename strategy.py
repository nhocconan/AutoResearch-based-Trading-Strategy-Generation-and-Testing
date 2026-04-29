#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 12h EMA50 trend filter and volume confirmation
# Long: price breaks above Donchian(20) high AND price > 12h EMA50 AND volume > 2.0x 20-bar avg
# Short: price breaks below Donchian(20) low AND price < 12h EMA50 AND volume > 2.0x 20-bar avg
# Exit: price crosses 12h EMA50 OR ATR stoploss (2.5 * ATR) OR opposite Donchian breakout
# Donchian provides clear structure for breakouts in both trending and ranging markets
# 12h EMA50 filters for higher timeframe trend alignment, reducing counter-trend trades
# Volume spike confirms institutional participation and reduces false breakouts
# Discrete position sizing: 0.28 for long/short to balance return and drawdown
# Target: 80-150 total trades over 4 years (20-38/year) on 4h timeframe

name = "4h_Donchian_Breakout_12hEMA50_VolumeSpike_ATRStop_v1"
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
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    donchian_high = rolling_max(high, 20)
    donchian_low = rolling_min(low, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_12h = ema_50_12h_aligned[i]
        curr_atr = atr[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 2.0 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Stoploss: 2.5 * ATR below entry
            stop_price = entry_price - 2.5 * curr_atr
            # Exit conditions: price below 12h EMA50 OR stoploss hit OR price breaks below Donchian low
            if curr_close < curr_ema_12h or curr_close < stop_price or curr_low < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
                
        elif position == -1:  # Short position
            # Stoploss: 2.5 * ATR above entry
            stop_price = entry_price + 2.5 * curr_atr
            # Exit conditions: price above 12h EMA50 OR stoploss hit OR price breaks above Donchian high
            if curr_close > curr_ema_12h or curr_close > stop_price or curr_high > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high AND price > 12h EMA50 AND volume spike
            if curr_high > donchian_high[i] and curr_close > curr_ema_12h and vol_spike:
                signals[i] = 0.28
                position = 1
                entry_price = curr_close
            # Short entry: price breaks below Donchian low AND price < 12h EMA50 AND volume spike
            elif curr_low < donchian_low[i] and curr_close < curr_ema_12h and vol_spike:
                signals[i] = -0.28
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals