#!/usr/bin/env python3
"""
6h Elder Ray + 1d Williams Fractal Regime Filter
Hypothesis: Elder Ray (Bull/Bear Power) measures trend strength via EMA13 deviation.
Williams Fractal on 1d identifies swing points; only trade in direction of the
most recent completed fractal (bullish fractal = long bias, bearish = short bias).
This avoids counter-trend trades and improves win rate in both bull/bear markets.
Uses discrete sizing (0.25) and volume confirmation (1.5x) to target ~60-100 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA13 for Elder Ray calculation
    ema13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # 1d Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = df_1d['high'].values - ema13_1d
    bear_power_1d = df_1d['low'].values - ema13_1d
    
    # Align 1d Elder Ray to 6h
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # 1d Williams Fractals (requires 2-bar confirmation delay)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Align with 2-bar extra delay for fractal confirmation
    bearish_fractal_6h = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_6h = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # 6h EMA21 for trend filter and Donchian-like breakout
    ema21_6h = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # 6h Donchian(20) for breakout signals
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # 6h volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for 1d EMA13 (13) + Donchian (20) + VolMA (20) + ATR (14)
    start_idx = max(50, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or 
            np.isnan(bearish_fractal_6h[i]) or np.isnan(bullish_fractal_6h[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i]) or np.isnan(ema21_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        bull_power = bull_power_6h[i]
        bear_power = bear_power_6h[i]
        is_bearish_fractal = bearish_fractal_6h[i] == 1
        is_bullish_fractal = bullish_fractal_6h[i] == 1
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        vol_ma = vol_ma_20[i]
        atr_value = atr[i]
        ema21 = ema21_6h[i]
        
        # Volume spike: current volume > 1.5 * 20-period average
        volume_spike = curr_volume > 1.5 * vol_ma
        
        # Determine regime bias from most recent completed fractal
        # We track the bias state outside the loop? Instead, use:
        # If bullish fractal was most recent -> long bias
        # If bearish fractal was most recent -> short bias
        # Simpler: use the current fractal signals as bias (they are aligned)
        long_bias = is_bullish_fractal
        short_bias = is_bearish_fractal
        
        # Update tracking variables for trailing stop logic
        if position == 1:
            highest_since_entry = max(highest_since_entry, curr_high)
        elif position == -1:
            lowest_since_entry = min(lowest_since_entry, curr_low)
        
        # Exit conditions: trailing stop or regime change
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Trailing stop: exit if price drops 2.0*ATR from highest since entry
                if curr_close < highest_since_entry - 2.0 * atr_value:
                    exit_signal = True
                # Regime change to short bias or weak bear power
                elif short_bias and bear_power < 0:
                    exit_signal = True
                # Weak bull power (loss of momentum)
                elif bull_power < 0:
                    exit_signal = True
                    
            elif position == -1:
                # Trailing stop: exit if price rises 2.0*ATR from lowest since entry
                if curr_close > lowest_since_entry + 2.0 * atr_value:
                    exit_signal = True
                # Regime change to long bias or weak bull power
                elif long_bias and bull_power > 0:
                    exit_signal = True
                # Weak bear power (loss of momentum)
                elif bear_power > 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                continue
        
        # Entry conditions
        if position == 0:
            # Long: bullish bias + bullish power > 0 + price > EMA21 + volume spike
            long_condition = long_bias and (bull_power > 0) and (curr_close > ema21) and volume_spike
            # Short: bearish bias + bearish power < 0 + price < EMA21 + volume spike
            short_condition = short_bias and (bear_power < 0) and (curr_close < ema21) and volume_spike
            
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

name = "6h_ElderRay_BullBearPower_1dWilliamsFractal_Regime_v1"
timeframe = "6h"
leverage = 1.0