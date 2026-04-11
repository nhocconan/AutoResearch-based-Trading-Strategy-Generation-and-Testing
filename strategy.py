#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d chop regime filter
# - Long: price breaks above Donchian upper channel (20-period high) + volume > 1.5x 20-period avg + 1d chop > 61.8 (range)
# - Short: price breaks below Donchian lower channel (20-period low) + volume > 1.5x 20-period avg + 1d chop > 61.8 (range)
# - Exit: price returns to Donchian midpoint (10-period average of high/low) or ATR stop (2.0 ATR)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits
# - Donchian breakouts capture strong momentum moves
# - Volume confirmation ensures institutional participation
# - Chop regime filter (>61.8) ensures we only trade in ranging markets where breakouts are meaningful

name = "4h_1d_donchian_volume_chop_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load 1d data ONCE before loop for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d Chopiness Index(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr_1d = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    
    # Sum of TR over 14 periods
    tr_sum_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chopiness Index = 100 * log10(tr_sum_14 / (hh_14 - ll_14)) / log10(14)
    # Avoid division by zero and log of zero
    range_14 = hh_14 - ll_14
    chop_raw = np.where((tr_sum_14 > 0) & (range_14 > 0), tr_sum_14 / range_14, 1.0)
    chop = 100 * np.log10(chop_raw) / np.log10(14)
    chop = np.where(np.isnan(chop), 50.0, chop)  # neutral value when undefined
    
    # Align 1d chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Pre-compute Donchian channels (20-period) on 4h timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Pre-compute volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute ATR for stoploss (4h timeframe)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(volume_sma_20[i]) or
            np.isnan(atr_14[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Donchian levels
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        midpoint = donchian_mid[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Regime filter: 1d chop > 61.8 (ranging market)
        chop_filter = chop_aligned[i] > 61.8
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price closes above upper Donchian channel
        if close_price > upper_channel and vol_confirm and chop_filter:
            enter_long = True
        
        # Short breakout: price closes below lower Donchian channel
        if close_price < lower_channel and vol_confirm and chop_filter:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to midpoint or ATR-based stop
            exit_long = (close_price <= midpoint) or (close_price <= entry_price - 2.0 * atr_14[i])
        elif position == -1:
            # Exit short if price returns to midpoint or ATR-based stop
            exit_short = (close_price >= midpoint) or (close_price >= entry_price + 2.0 * atr_14[i])
        
        # Track entry price for stoploss calculation
        if enter_long or enter_short:
            entry_price = close_price
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals