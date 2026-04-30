#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
# Uses weekly Camarilla pivot levels (R4/S4) from 1w data to determine major trend direction.
# Only takes breakouts in the direction of the weekly pivot bias to avoid counter-trend trades.
# Volume confirmation (1.5x 20-period average) filters low-quality breakouts.
# Designed for low trade frequency (target: 80-120 total trades over 4 years) to avoid fee drag.
# Weekly pivot filter provides structural bias that works in both bull and bear markets by
# aligning with higher-timeframe support/resistance levels.

name = "6h_Donchian20_1wCamarillaPivot_VolumeConfirm_v1"
timeframe = "6h"
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
    
    # Load 1w data ONCE before loop for weekly Camarilla pivot trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels (based on prior week's OHLC)
    # Camarilla: Pivot = (H+L+C)/3, Range = H-L
    # R4 = C + Range * 1.1/2, S4 = C - Range * 1.1/2
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_range = weekly_high - weekly_low
    camarilla_r4 = weekly_close + (weekly_range * 1.1 / 2.0)
    camarilla_s4 = weekly_close - (weekly_range * 1.1 / 2.0)
    
    # Determine weekly bias: price above R4 = bullish bias, below S4 = bearish bias
    # We use the prior week's levels to avoid look-ahead
    weekly_bias_bullish = np.roll(camarilla_r4, 1)  # Prior week's R4
    weekly_bias_bearish = np.roll(camarilla_s4, 1)  # Prior week's S4
    weekly_bias_bullish[0] = np.nan  # First value has no prior week
    weekly_bias_bearish[0] = np.nan
    
    # Align weekly bias to 6h timeframe (only use completed weekly bars)
    weekly_bias_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias_bullish)
    weekly_bias_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias_bearish)
    
    # ATR(14) for volatility and stoploss
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average (moderate threshold)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(1, 20, atr_period, 20) + 1  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(weekly_bias_bullish_aligned[i]) or
            np.isnan(weekly_bias_bearish_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_volume_confirm = volume_confirm[i]
        curr_weekly_bullish = weekly_bias_bullish_aligned[i]
        curr_weekly_bearish = weekly_bias_bearish_aligned[i]
        
        # Calculate 6h Donchian channels using only completed 6h bars
        if i >= 20:
            donchian_high = np.max(high[i-20:i])
            donchian_low = np.min(low[i-20:i])
        else:
            donchian_high = np.nan
            donchian_low = np.nan
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper channel, weekly bullish bias (price > weekly R4), volume confirmation
            if (curr_close > donchian_high and 
                curr_close > curr_weekly_bullish and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
            # Short: price breaks below Donchian lower channel, weekly bearish bias (price < weekly S4), volume confirmation
            elif (curr_close < donchian_low and 
                  curr_close < curr_weekly_bearish and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                lowest_since_entry = curr_close
        
        elif position == 1:  # Long position
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # ATR trailing stop: exit if price drops 2.5*ATR from highest point (wider stop for 6h)
            if curr_close < highest_since_entry - (2.5 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # ATR trailing stop: exit if price rises 2.5*ATR from lowest point
            if curr_close > lowest_since_entry + (2.5 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals