#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA50 trend filter and volume confirmation
# Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# Long when Bull Power > 0 AND Bear Power increasing (less negative) AND price > 1d EMA50 AND volume > 1.5x 20-bar avg
# Short when Bear Power < 0 AND Bull Power decreasing (less positive) AND price < 1d EMA50 AND volume > 1.5x 20-bar avg
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year on 6h.
# Works in bull markets by capturing buying strength, works in bear by identifying selling exhaustion.

name = "6h_ElderRay_1dEMA50_Trend_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d close
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate EMA(13) for Elder Ray on 6h close (primary timeframe)
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull Power: High - EMA(13)
    bear_power = low - ema_13   # Bear Power: Low - EMA(13)
    
    # Calculate momentum of Elder Ray (change from previous bar)
    bull_power_momentum = bull_power - np.roll(bull_power, 1)
    bear_power_momentum = bear_power - np.roll(bear_power, 1)
    # Set first bar momentum to 0 (no previous bar)
    bull_power_momentum[0] = 0
    bear_power_momentum[0] = 0
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: >1.5x 20-bar average volume (moderate filter)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 13)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_trend = ema_50_1d_aligned[i]
        curr_close = close[i]
        curr_bull = bull_power[i]
        curr_bear = bear_power[i]
        bull_mom = bull_power_momentum[i]
        bear_mom = bear_power_momentum[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when Bull Power > 0 AND Bear Power increasing (less negative) 
            # AND price > 1d EMA50 AND volume confirmation
            if (curr_bull > 0 and bear_mom > 0 and curr_close > ema_trend and vol_conf):
                signals[i] = 0.25
                position = 1
            # Short when Bear Power < 0 AND Bull Power decreasing (less positive)
            # AND price < 1d EMA50 AND volume confirmation
            elif (curr_bear < 0 and bull_mom < 0 and curr_close < ema_trend and vol_conf):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when Bull Power turns negative OR Bear Power > 0
            if curr_bull <= 0 or curr_bear >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when Bear Power turns positive OR Bull Power < 0
            if curr_bear >= 0 or curr_bull <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals