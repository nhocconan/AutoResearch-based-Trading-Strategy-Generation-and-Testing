#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter, volume confirmation, and ATR-based stoploss.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for EMA50 trend direction and volume spike filter.
- Donchian Channel(20): Upper/lower bands from 20-period high/low.
- Trend Filter: Price > EMA50(1d) for long bias, Price < EMA50(1d) for short bias.
- Volume Confirmation: Current volume > 1.8 * 20-period average volume (1d).
- Entry: Long when close crosses above Upper Band AND price > EMA50 AND volume confirmation.
         Short when close crosses below Lower Band AND price < EMA50 AND volume confirmation.
- Exit: Opposite Donchian breakout (long exits on close < Lower Band, short exits on close > Upper Band).
- Stoploss: ATR-based (2 * ATR(14)) - implemented via signal=0 when stop level breached on close.
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by aligning with 1d trend and filtering breakouts with volume.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate ATR(14) for stoploss
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Calculate Donchian Channel(20)
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(donchian_period, 50, atr_period)  # Need 20 for Donchian, 50 for EMA50, 14 for ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Trend filter: price > EMA50 for long bias, price < EMA50 for short bias
        long_bias = curr_close > ema50_1d_aligned[i]
        short_bias = curr_close < ema50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.8 * 20-period average volume
        volume_confirm = curr_volume > 1.8 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Donchian levels
        upper_band = highest_high[i]
        lower_band = lowest_low[i]
        
        # ATR-based stoploss level
        atr_value = atr[i]
        stoploss_long = entry_price - 2.0 * atr_value if position == 1 else np.nan
        stoploss_short = entry_price + 2.0 * atr_value if position == -1 else np.nan
        
        # Check stoploss hit (using close price)
        if position != 0:
            if position == 1 and not np.isnan(stoploss_long) and curr_close <= stoploss_long:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
            elif position == -1 and not np.isnan(stoploss_short) and curr_close >= stoploss_short:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # Exit conditions: opposite Donchian breakout
        if position != 0:
            if position == 1 and curr_close < lower_band:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
            elif position == -1 and curr_close > upper_band:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # Entry conditions: Donchian breakout with trend and volume filters
        if position == 0:
            # Long: close crosses above Upper Band AND long bias AND volume confirmation
            long_condition = (curr_close > upper_band) and (close[i-1] <= upper_band if i > 0 else False) and long_bias and volume_confirm
            
            # Short: close crosses below Lower Band AND short bias AND volume confirmation
            short_condition = (curr_close < lower_band) and (close[i-1] >= lower_band if i > 0 else False) and short_bias and volume_confirm
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA50Trend_VolumeConfirm_ATRStop_v1"
timeframe = "4h"
leverage = 1.0