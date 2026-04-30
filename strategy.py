#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Uses 1d EMA50 for long-term trend alignment to reduce whipsaws in both bull and bear markets.
# Volume > 1.8x 20-period average confirms strong momentum (moderate threshold to control trade frequency).
# ATR-based stoploss (2.0x) manages risk. Designed for low trade frequency (~15-25 trades/year) to minimize fee drag on 12h timeframe.
# Entry requires 1d EMA50 alignment + volume spike + Donchian breakout.

name = "12h_Donchian20_1dEMA50_VolumeConfirm_ATRStop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 55:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d data
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for 12h timeframe stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period) for breakout signals
    # Highest high over last 20 periods
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lowest low over last 20 periods
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 100  # warmup for EMA, ATR and Donchian
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_50_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema = ema_50_aligned[i]
        curr_atr = atr[i]
        curr_highest_high = highest_high[i]
        curr_lowest_low = lowest_low[i]
        
        # Volume confirmation: volume > 1.8x 20-period average (moderate threshold to control trades)
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > (1.8 * vol_ma_20)
        else:
            volume_confirm = False
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper channel, price above 1d EMA50, volume spike
            if (curr_close > curr_highest_high and 
                curr_close > curr_ema and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below Donchian lower channel, price below 1d EMA50, volume spike
            elif (curr_close < curr_lowest_low and 
                  curr_close < curr_ema and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit conditions: price breaks below Donchian lower channel OR stoploss hit
            if (curr_close < curr_lowest_low or 
                curr_close < entry_price - 2.0 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: price breaks above Donchian upper channel OR stoploss hit
            if (curr_close > curr_highest_high or 
                curr_close > entry_price + 2.0 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals