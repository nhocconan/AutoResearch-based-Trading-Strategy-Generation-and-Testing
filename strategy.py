#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 12h EMA34 trend + volume spike confirmation + ATR stoploss.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h for EMA34 trend filter to capture intermediate trend direction.
- Donchian(20): Price channel breakout identifies momentum bursts.
- Entry: Long when price breaks above Donchian(20) upper band AND price > 12h EMA34 AND volume > 2.0 * 20-period average volume.
         Short when price breaks below Donchian(20) lower band AND price < 12h EMA34 AND volume > 2.0 * 20-period average volume.
- Exit: ATR-based trailing stop (3 * ATR(14)) OR opposite Donchian breakout.
- Signal size: 0.30 discrete to balance profit potential and fee drag.
- Donchian breakouts capture strong moves, effective in both trending and ranging markets when combined with trend filter.
- 12h EMA34 provides intermediate trend filter to avoid counter-trend trades during corrections.
- Volume spike confirmation ensures breakouts have participation, reducing false signals.
- ATR stoploss manages risk during adverse moves.
- Estimated trades: ~100 total over 4 years (~25/year) based on Donchian breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def atr(high, low, close, period=14):
    """Calculate Average True Range."""
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    high_close[0] = high_low[0]
    low_close[0] = high_low[0]
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    return pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values

def donchian_channels(high, low, period=20):
    """Calculate Donchian channels."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need sufficient data for indicators
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h trend filter: EMA34
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 35:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    ema34_12h = ema(df_12h['close'].values, 34)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h, additional_delay_bars=1)
    
    # Calculate 12h volume average for confirmation
    if len(df_12h) < 21:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = df_12h['volume'].values / (vol_ma_20 + 1e-10)  # Avoid division by zero
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h, additional_delay_bars=1)
    
    # Donchian channels (20-period)
    upper_band, lower_band = donchian_channels(high, low, 20)
    
    # ATR for stoploss
    atr_val = atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start from index where all indicators are ready
    start_idx = 100  # Need sufficient data for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ratio_12h_aligned[i]) or
            np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(atr_val[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Update highest/lowest since entry for trailing stop
        if position == 1:
            highest_since_entry = max(highest_since_entry, curr_high)
        elif position == -1:
            lowest_since_entry = min(lowest_since_entry, curr_low)
        
        # Check stoploss conditions
        if position != 0:
            stop_triggered = False
            if position == 1:
                # Long stop: price drops below highest_since_entry - 3 * ATR
                if curr_close < highest_since_entry - 3.0 * atr_val[i]:
                    stop_triggered = True
            elif position == -1:
                # Short stop: price rises above lowest_since_entry + 3 * ATR
                if curr_close > lowest_since_entry + 3.0 * atr_val[i]:
                    stop_triggered = True
            
            if stop_triggered:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                continue
        
        # Check opposite Donchian breakout for exit
        if position != 0:
            exit_signal = False
            if position == 1 and curr_close < lower_band[i]:
                exit_signal = True
            elif position == -1 and curr_close > upper_band[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                continue
        
        # Entry conditions: Donchian breakout with trend filter and volume confirmation
        if position == 0:
            # Volume confirmation: current volume > 2.0 * 20-period average volume
            vol_ma_20_current = vol_ma_20[min(i, len(vol_ma_20)-1)] if len(vol_ma_20) > 0 else 0
            volume_confirm = curr_volume > 2.0 * vol_ma_20_current
            
            # Long: Price breaks above upper Donchian band AND price > 12h EMA34 AND volume confirmation
            if curr_close > upper_band[i] and curr_close > ema34_12h_aligned[i] and volume_confirm:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
            # Short: Price breaks below lower Donchian band AND price < 12h EMA34 AND volume confirmation
            elif curr_close < lower_band[i] and curr_close < ema34_12h_aligned[i] and volume_confirm:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.30
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA34_TrendFilter_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0