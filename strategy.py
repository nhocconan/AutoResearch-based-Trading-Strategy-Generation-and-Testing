#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band squeeze breakout with 1d trend filter (EMA34) and volume confirmation (volume > 1.5x 20-day avg)
# Long when price breaks above upper BB after squeeze + price > daily EMA34 + volume > 1.5x 20-day avg
# Short when price breaks below lower BB after squeeze + price < daily EMA34 + volume > 1.5x 20-day avg
# Exit when price returns to middle BB (20-day SMA)
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag

name = "4h_Bollinger_Squeeze_Breakout_1dEMA34_Volume"
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
    
    # Get daily data for trend filter and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate daily 20-day volume average for volume filter
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align daily indicators to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate Bollinger Bands on 4h data (20-period, 2 std)
    close_series = pd.Series(close)
    sma_20 = close_series.rolling(window=20, min_periods=20).mean()
    std_20 = close_series.rolling(window=20, min_periods=20).std()
    upper_bb = (sma_20 + 2 * std_20).values
    lower_bb = (sma_20 - 2 * std_20).values
    middle_bb = sma_20.values
    
    # Bollinger Band width for squeeze detection (normalized by middle BB)
    bb_width = ((upper_bb - lower_bb) / middle_bb) * 100
    # Squeeze condition: BB width below 20-period average of BB width
    bb_width_ma_20 = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze_condition = bb_width < bb_width_ma_20
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or \
           np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or np.isnan(middle_bb[i]) or \
           np.isnan(squeeze_condition[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 4h volume > 1.5x 20-day average volume
        vol_filter = volume[i] > 1.5 * vol_ma_20_aligned[i] if not np.isnan(vol_ma_20_aligned[i]) else False
        
        if position == 0:
            # Look for entry: BB breakout after squeeze + trend + volume
            long_condition = squeeze_condition[i-1] and close[i] > upper_bb[i] and close[i] > ema_34_aligned[i] and vol_filter
            short_condition = squeeze_condition[i-1] and close[i] < lower_bb[i] and close[i] < ema_34_aligned[i] and vol_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to middle BB
            if close[i] <= middle_bb[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to middle BB
            if close[i] >= middle_bb[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals