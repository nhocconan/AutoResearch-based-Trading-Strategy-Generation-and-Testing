#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot direction filter and volume confirmation
# Long when price breaks above Donchian(20) high, 1d price above weekly pivot (PP), volume > 2.0x 20-bar average
# Short when price breaks below Donchian(20) low, 1d price below weekly pivot (PP), volume > 2.0x 20-bar average
# Uses 6h timeframe targeting 12-37 trades/year (~50-150 total over 4 years) to minimize fee drag.
# Works in bull markets via Donchian breakouts and in bear markets via breakdowns with weekly pivot bias.

name = "6h_Donchian20_1dWeeklyPivot_Direction_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 5:  # Need at least a week of data for weekly pivot
        return np.zeros(n)
    
    # Calculate weekly pivot points from 1d data (using last completed week)
    # Weekly PP = (Weekly High + Weekly Low + Weekly Close) / 3
    # We'll use rolling weekly window on daily data
    weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().values  # 5 trading days
    weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().values
    weekly_pp = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1d, weekly_pp)
    
    # Calculate 6h Donchian channels (20-bar)
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(donchian_window, 20, 5)  # Donchian20, volume MA20, weekly calc
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pp_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        price = close[i]
        curr_weekly_pp = weekly_pp_aligned[i]
        upper_band = donchian_high[i]
        lower_band = donchian_low[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian high, price above weekly PP, volume spike
            if price > upper_band and price > curr_weekly_pp and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: Price breaks below Donchian low, price below weekly PP, volume spike
            elif price < lower_band and price < curr_weekly_pp and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on stoploss or breakdown below Donchian low
            # ATR-based stoploss: 2.5 * ATR below entry (using 6h ATR)
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price - 2.5 * atr_val
            # Exit on stoploss or when price breaks below Donchian low (trend reversal)
            if price < stop_loss or price < lower_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on stoploss or breakout above Donchian high
            # ATR-based stoploss: 2.5 * ATR above entry
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price + 2.5 * atr_val
            # Exit on stoploss or when price breaks above Donchian high (trend reversal)
            if price > stop_loss or price > upper_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals