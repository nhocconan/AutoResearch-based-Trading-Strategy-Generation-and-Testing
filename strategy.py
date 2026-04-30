#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and ATR-based volume confirmation.
# Long when price breaks above upper Donchian channel, price > 1d EMA50, and volume > 1.5x ATR-scaled average volume.
# Short when price breaks below lower Donchian channel, price < 1d EMA50, and volume > 1.5x ATR-scaled average volume.
# Exit when price crosses the 10-period EMA (trend reversal signal) or ATR-based stoploss is hit.
# Uses 1d EMA50 for higher timeframe trend alignment and ATR-scaled volume to avoid low-volatility false breakouts.
# Targets 20-50 trades/year on 4h timeframe with discrete position sizing (0.25) to minimize fee drag.
# Works in bull markets via trend-aligned breakouts and in bear markets via short breakdowns with trend filter.

name = "4h_Donchian20_1dEMA50_ATRVolConfirm_Trend_v1"
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
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for volume confirmation scaling and stoploss
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR-scaled average volume (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    atr_scaled_vol_ma = vol_ma_20 * (atr / np.nanmean(atr))  # normalize by average ATR
    volume_confirm = volume > (1.5 * atr_scaled_vol_ma)
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 10-period EMA for exit signal
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_10[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_highest_high = highest_high[i]
        curr_lowest_low = lowest_low[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_ema_10 = ema_10[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper Donchian, price > 1d EMA50, volume confirmation
            if (curr_close > curr_highest_high and 
                curr_close > curr_ema_50_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian, price < 1d EMA50, volume confirmation
            elif (curr_close < curr_lowest_low and 
                  curr_close < curr_ema_50_1d and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit conditions: price crosses below 10 EMA (trend reversal) or ATR stoploss
            if curr_close < curr_ema_10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: price crosses above 10 EMA (trend reversal) or ATR stoploss
            if curr_close > curr_ema_10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals