#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and ATR-based volume confirmation.
# Long when price breaks above Donchian upper band AND 12h EMA50 uptrend AND volume > 1.5x ATR-scaled volume median.
# Short when price breaks below Donchian lower band AND 12h EMA50 downtrend AND volume > 1.5x ATR-scaled volume median.
# Uses ATR to normalize volume for volatility regimes, reducing false breakouts in low-volatility environments.
# Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
# Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years) to minimize fee drag.

name = "4h_Donchian20_Breakout_12hEMA50_ATRVol_v1"
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
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for volume normalization
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR-scaled volume median (20-period)
    vol_atr_ratio = volume / (atr_14 + 1e-10)  # Avoid division by zero
    vol_atr_median_20 = pd.Series(vol_atr_ratio).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for all indicators
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or 
            np.isnan(vol_atr_median_20[i]) or 
            np.isnan(atr_14[i]) or atr_14[i] <= 0):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_atr = atr_14[i]
        
        # Trend filter: 12h EMA50 direction
        uptrend = curr_close > ema_50_12h_aligned[i]
        downtrend = curr_close < ema_50_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.5x ATR-scaled 20-period volume median
        if vol_atr_median_20[i] <= 0 or np.isnan(vol_atr_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_atr_median_20[i] * curr_atr * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper band AND uptrend AND volume spike
            if curr_close > highest_20[i] and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below Donchian lower band AND downtrend AND volume spike
            elif curr_close < lowest_20[i] and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls back below Donchian upper band OR trend turns down
            if curr_close < highest_20[i] or not uptrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises back above Donchian lower band OR trend turns up
            if curr_close > lowest_20[i] or not downtrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals