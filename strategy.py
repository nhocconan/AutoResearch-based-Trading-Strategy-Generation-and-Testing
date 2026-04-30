#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d EMA34 trend filter + volume spike confirmation.
# Uses 1d EMA34 for stable daily trend direction and requires volume > 2.0x 20-period average.
# Takes long when price breaks above Donchian(20) high in uptrend, short when breaks below Donchian(20) low in downtrend.
# Added ATR-based stoploss (2.0x ATR) and time-based exit (max 3 bars holding).
# Designed for low trade frequency (~15-25 trades/year) to minimize fee drag and avoid overtrading.
# Donchian channels provide clear structure that works in both trending and ranging markets when combined with trend filter.

name = "6h_Donchian20_1dEMA34_VolumeSpike_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR for stoploss (using 14-period ATR on 6h)
    if n >= 14:
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    else:
        atr = np.full(n, np.nan)
    
    # Calculate Donchian(20) channels on 6h
    if n >= 20:
        # Donchian high: highest high of last 20 periods
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        # Donchian low: lowest low of last 20 periods
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_in_trade = 0  # track holding period for time-based exit
    
    start_idx = 50  # warmup for EMA34 and Donchian
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            not in_session[i]):
            signals[i] = 0.0
            bars_in_trade = 0  # reset counter when flat
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_atr = atr[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        
        # Volume confirmation: volume > 2.0x 20-period average (moderate to balance frequency)
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > (2.0 * vol_ma_20)
        else:
            volume_confirm = False
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian high, 1d EMA34 uptrend, volume spike
            if (curr_close > curr_donchian_high and 
                curr_close > curr_ema_34_1d and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                bars_in_trade = 1
            # Short: price breaks below Donchian low, 1d EMA34 downtrend, volume spike
            elif (curr_close < curr_donchian_low and 
                  curr_close < curr_ema_34_1d and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                bars_in_trade = 1
        
        elif position == 1:  # Long position
            bars_in_trade += 1
            # Exit conditions: price breaks below Donchian low, ATR stoploss hit, or max 3 bars held
            if (curr_close < curr_donchian_low) or \
               (curr_close < entry_price - 2.0 * curr_atr) or \
               (bars_in_trade >= 3):
                signals[i] = 0.0
                position = 0
                bars_in_trade = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            bars_in_trade += 1
            # Exit conditions: price breaks above Donchian high, ATR stoploss hit, or max 3 bars held
            if (curr_close > curr_donchian_high) or \
               (curr_close > entry_price + 2.0 * curr_atr) or \
               (bars_in_trade >= 3):
                signals[i] = 0.0
                position = 0
                bars_in_trade = 0
            else:
                signals[i] = -0.25
    
    return signals