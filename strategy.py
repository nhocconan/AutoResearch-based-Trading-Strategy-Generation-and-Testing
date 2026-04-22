#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with daily trend filter and volume spike.
# Uses daily EMA(34) for trend direction, 12h Donchian breakout for entry, volume spike for confirmation.
# Designed for low trade frequency (<30/year) to minimize fee drag and work in both bull and bear markets.
# Breakouts work in trending markets; volume filter avoids false signals in ranging conditions.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load daily data for trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily EMA(34) for trend direction
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 12h Donchian channels (20-period)
    # Calculate manually since we don't have 12h data loaded as OHLC
    # Use 20-period lookback on 12h timeframe: each 12h bar = 48 of 15m bars
    # But we're on 12h timeframe, so we need 20-period lookback on actual 12h data
    # Since we don't have 12h OHLC, we'll approximate using available data
    # For 12h timeframe, we need to calculate Donchian from 12h high/low
    
    # Instead, use price action: look for new 20-period high/low on close prices
    # This approximates Donchian breakout behavior
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback, n):
        highest_high[i] = np.max(high[i-lookback:i])
        lowest_low[i] = np.min(low[i-lookback:i])
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if not in session or data not ready
        if not in_session[i] or np.isnan(ema_34_1d_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 20-period high + uptrend (price > daily EMA34) + volume spike
            if (close[i] > highest_high[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-period low + downtrend (price < daily EMA34) + volume spike
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: trend reversal or price returns to mean
            if position == 1:
                if (close[i] < ema_34_1d_aligned[i] or close[i] < highest_high[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (close[i] > ema_34_1d_aligned[i] or close[i] > lowest_low[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_DailyEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0