#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Uses discrete sizing 0.25 to balance return and drawdown. Target: 100-180 total trades over 4 years (25-45/year).
# Donchian breakout captures momentum; 12h EMA50 ensures alignment with intermediate trend.
# Volume spike filters for institutional participation. Works in bull via breakout longs, in bear via breakdown shorts.
# ATR-based stoploss limits drawdown during adverse moves.

name = "4h_Donchian20_12hEMA50_VolumeSpike_ATRStop_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h EMA(50) for trend filter (MTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop = 0.0
    
    start_idx = max(20, 50, 20, 14)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_highest_20 = highest_20[i]
        curr_lowest_20 = lowest_20[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_volume_spike = volume_spike[i]
        curr_atr = atr[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: price breaks above Donchian upper AND above 12h EMA50 (bullish bias)
                if (curr_close > curr_highest_20 and 
                    curr_close > curr_ema_50_12h):
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    atr_stop = entry_price - 2.5 * curr_atr  # initial stop
                # Bearish entry: price breaks below Donchian lower AND below 12h EMA50 (bearish bias)
                elif (curr_close < curr_lowest_20 and 
                      curr_close < curr_ema_50_12h):
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    atr_stop = entry_price + 2.5 * curr_atr  # initial stop
        
        elif position == 1:  # Long position
            # Update trailing stop: move stop up as price makes new highs
            if curr_high > entry_price:
                atr_stop = max(atr_stop, curr_high - 2.5 * curr_atr)
            # Exit: price hits trailing stop OR breaks below 12h EMA50 (trend change)
            if (curr_low <= atr_stop or 
                curr_close < curr_ema_50_12h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update trailing stop: move stop down as price makes new lows
            if curr_low < entry_price:
                atr_stop = min(atr_stop, curr_low + 2.5 * curr_atr)
            # Exit: price hits trailing stop OR breaks above 12h EMA50 (trend change)
            if (curr_high >= atr_stop or 
                curr_close > curr_ema_50_12h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals