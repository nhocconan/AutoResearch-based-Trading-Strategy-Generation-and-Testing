#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter (price > SMA50) and volume confirmation.
# Long when price breaks above Donchian high(20) AND price > 1d SMA50 AND volume > 2.0x 96-bar average.
# Short when price breaks below Donchian low(20) AND price < 1d SMA50 AND volume > 2.0x 96-bar average.
# Uses discrete sizing 0.25 to limit drawdown. Session filter 08-20 UTC to avoid low-liquidity hours.
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
# 1d SMA50 provides robust trend alignment that works in both bull (price above SMA) and bear (price below SMA).
# Donchian(20) offers reliable breakout points with proven edge on SOLUSDT.
# Volume confirmation (2.0x average) ensures only high-conviction breakouts are traded.

name = "4h_Donchian20_1dSMA50_Trend_VolumeConfirm_v1"
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
    open_time = prices['open_time']
    
    # Pre-compute session hours for efficiency (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load 1d data ONCE before loop for SMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d SMA50 calculation
    close_1d = df_1d['close'].values
    sma_50 = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma_50_aligned = align_htf_to_ltf(prices, df_1d, sma_50)
    
    # 1d trend: price above/below SMA50
    price_above_sma = close > sma_50_aligned
    price_below_sma = close < sma_50_aligned
    
    # Donchian(20) on 4h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current 4h volume > 2.0x 96-bar average (approx 16d on 4h)
    vol_ma = pd.Series(volume).rolling(window=96, min_periods=96).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Session filter: trade only 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(sma_50_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)
        
        # Donchian breakout signals
        breakout_up = curr_high > donchian_high[i-1]  # break above previous Donchian high
        breakout_down = curr_low < donchian_low[i-1]  # break below previous Donchian low
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above Donchian high AND price > 1d SMA50 AND volume confirmation
            if (breakout_up and 
                price_above_sma[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low AND price < 1d SMA50 AND volume confirmation
            elif (breakout_down and 
                  price_below_sma[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below Donchian low (stoploss) OR price < 1d SMA50 (trend change)
            if (curr_low < donchian_low[i] or 
                not price_above_sma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian high (stoploss) OR price > 1d SMA50 (trend change)
            if (curr_high > donchian_high[i] or 
                not price_below_sma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals