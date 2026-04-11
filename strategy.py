#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume confirmation + 1h chop regime filter
# - Long: price breaks above Donchian(20) high + 1d volume > 1.5x 20-period avg + 1h chop < 61.8 (trending)
# - Short: price breaks below Donchian(20) low + 1d volume > 1.5x 20-period avg + 1h chop < 61.8 (trending)
# - Exit: price returns to Donchian(20) midpoint or ATR-based stop (2.0 ATR)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits
# - Donchian breakouts capture strong momentum moves
# - Volume confirmation ensures institutional participation
# - Chop filter avoids whipsaw in ranging markets

name = "4h_1d_1h_donchian_volume_chop_v1"
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
    
    # Load 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d volume SMA(20)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Load 1h data ONCE before loop for chop regime filter
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 50:
        return signals
    
    # Pre-compute 1h Chopiness Index(14)
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # True Range
    tr_1h = np.maximum(high_1h - low_1h, np.maximum(np.abs(high_1h - np.roll(close_1h, 1)), np.abs(low_1h - np.roll(close_1h, 1))))
    tr_1h[0] = high_1h[0] - low_1h[0]
    
    # Sum of TR over 14 periods
    tr_sum_14 = pd.Series(tr_1h).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1h).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1h).rolling(window=14, min_periods=14).min().values
    
    # Chopiness Index = 100 * log10(tr_sum_14 / (hh_14 - ll_14)) / log10(14)
    # Avoid division by zero and log of zero
    hl_range = hh_14 - ll_14
    chop_raw = np.where((hl_range > 0) & (tr_sum_14 > 0), 
                        100 * np.log10(tr_sum_14 / hl_range) / np.log10(14), 
                        50.0)  # neutral value when undefined
    chopiness = chop_raw
    
    # Align 1h chopiness to 4h timeframe
    chopiness_aligned = align_htf_to_ltf(prices, df_1h, chopiness)
    
    # Pre-compute ATR for stoploss (4h timeframe)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute Donchian channels (20-period) for 4h timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(volume_sma_20_aligned[i]) or np.isnan(chopiness_aligned[i]) or 
            np.isnan(atr_14[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Donchian levels
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        mid_channel = donchian_mid[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Regime filter: 1h chop < 61.8 (trending market)
        chop_filter = chopiness_aligned[i] < 61.8
        
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
            exit_long = (close_price <= mid_channel) or (close_price <= entry_price - 2.0 * atr_14[i])
        elif position == -1:
            # Exit short if price returns to midpoint or ATR-based stop
            exit_short = (close_price >= mid_channel) or (close_price >= entry_price + 2.0 * atr_14[i])
        
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