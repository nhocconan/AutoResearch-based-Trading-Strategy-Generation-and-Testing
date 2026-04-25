#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + 1d ATR Trend + Volume Spike
Hypothesis: On 12h timeframe, Donchian breakouts in direction of 1d ATR-based trend
with volume confirmation capture sustained moves while avoiding whipsaws.
ATR trend filter adapts to bull/bear regimes: long when price > close + 0.5*ATR(1d),
short when price < close - 0.5*ATR(1d). Uses discrete sizing (0.25) to target
~12-30 trades/year (50-150 over 4 years) for optimal fee efficiency.
Works in both bull/bear by following ATR-defined trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 1d trend: price > close + 0.5*ATR = uptrend, price < close - 0.5*ATR = downtrend
    trend_up_1d = close_1d > (close_1d + 0.5 * atr_1d)  # Simplified: close_1d > close_1d + 0.5*atr_1d is always false
    trend_down_1d = close_1d < (close_1d - 0.5 * atr_1d)  # Simplified: close_1d < close_1d - 0.5*atr_1d is always false
    
    # Correct trend calculation: compare current close to prior close ± ATR
    # Uptrend: current close > prior close + 0.5*ATR
    # Downtrend: current close < prior close - 0.5*ATR
    close_shift_1d = np.concatenate([[np.nan], close_1d[:-1]])
    atr_shift_1d = np.concatenate([[np.nan], atr_1d[:-1]])
    trend_up_1d = close_1d > (close_shift_1d + 0.5 * atr_shift_1d)
    trend_down_1d = close_1d < (close_shift_1d - 0.5 * atr_shift_1d)
    
    # Align 1d trend to 12h timeframe
    trend_up_12h = align_htf_to_ltf(prices, df_1d, trend_up_1d.astype(float))
    trend_down_12h = align_htf_to_ltf(prices, df_1d, trend_down_1d.astype(float))
    
    # 12h Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # 12h volume average for confirmation (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for dynamic stop (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for 1d trend (shifted) + Donchian (20) + VolMA (20) + ATR (14)
    start_idx = max(30, 20, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trend_up_12h[i]) or np.isnan(trend_down_12h[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        atr_value = atr[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma_20[i]
        
        # Donchian breakout conditions
        bullish_breakout = curr_close > donchian_high[i]  # Break above upper band
        bearish_breakout = curr_close < donchian_low[i]   # Break below lower band
        
        # Update tracking variables for trailing stop logic
        if position == 1:
            highest_since_entry = max(highest_since_entry, curr_high)
        elif position == -1:
            lowest_since_entry = min(lowest_since_entry, curr_low)
        
        # Exit conditions: trailing stop or reverse breakout
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Trailing stop: exit if price drops 3.0*ATR from highest since entry
                if curr_close < highest_since_entry - 3.0 * atr_value:
                    exit_signal = True
                # Reverse breakout or trend reversal
                elif curr_close < donchian_low[i] or trend_down_12h[i] > 0.5:
                    exit_signal = True
                    
            elif position == -1:
                # Trailing stop: exit if price rises 3.0*ATR from lowest since entry
                if curr_close > lowest_since_entry + 3.0 * atr_value:
                    exit_signal = True
                # Reverse breakout or trend reversal
                elif curr_close > donchian_high[i] or trend_up_12h[i] > 0.5:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                continue
        
        # Entry conditions: Donchian breakout + trend alignment + volume
        if position == 0:
            # Long: break above Donchian high AND 1d uptrend AND volume spike
            long_condition = bullish_breakout and (trend_up_12h[i] > 0.5) and volume_spike
            # Short: break below Donchian low AND 1d downtrend AND volume spike
            short_condition = bearish_breakout and (trend_down_12h[i] > 0.5) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
            elif short_condition:
                signals[i] = -0.25
                position = -1
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dATR_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0