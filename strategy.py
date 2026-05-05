#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d EMA50 trend filter, volume spike confirmation, and ATR stoploss
# Long when price breaks above Donchian(20) high AND price > 1d EMA50 AND volume > 2.0 * avg_volume(20)
# Short when price breaks below Donchian(20) low AND price < 1d EMA50 AND volume > 2.0 * avg_volume(20)
# Exit via ATR(14) trailing stop: long when price < highest_high_since_entry - 2.5 * ATR, short when price > lowest_low_since_entry + 2.5 * ATR
# Uses discrete sizing 0.25 to control risk and minimize fee churn
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Donchian provides objective price channels, EMA50 filters trend direction, volume spike confirms breakout strength
# Works in bull markets (buying breakouts in uptrend) and bear markets (selling breakdowns in downtrend)
# ATR stoploss manages risk without look-ahead, using only close-based exits

name = "4h_Donchian20_1dEMA50_VolumeSpike_ATRStop"
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
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA50
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Donchian(20) channels on 4h
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for volatility and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i]) or np.isnan(atr[i]) or np.isnan(avg_volume_20[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high, above 1d EMA50, volume confirmation, in session
            if (close[i] > highest_high_20[i] and close[i] > ema50_1d_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
                highest_since_entry = high[i]  # Initialize trailing stop high
            # Short: price breaks below Donchian low, below 1d EMA50, volume confirmation, in session
            elif (close[i] < lowest_low_20[i] and close[i] < ema50_1d_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                lowest_since_entry = low[i]  # Initialize trailing stop low
        elif position == 1:
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, high[i])
            # Exit long: ATR trailing stop
            if close[i] < highest_since_entry - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, low[i])
            # Exit short: ATR trailing stop
            if close[i] > lowest_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals