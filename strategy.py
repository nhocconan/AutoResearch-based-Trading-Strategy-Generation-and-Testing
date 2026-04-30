#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA50 trend filter and volume confirmation.
# Long when Bull Power > 0 (close > EMA13), Bear Power < 0 (low < EMA13), 1d uptrend (close > 1d EMA50), and volume > 1.5x 20-bar avg.
# Short when Bear Power < 0 (low < EMA13), Bull Power < 0 (close < EMA13), 1d downtrend (close < 1d EMA50), and volume > 1.5x 20-bar avg.
# Exit when Bull Power <= 0 (for long) or Bear Power >= 0 (for short).
# Uses Elder Ray to measure bull/bear power relative to EMA13, filtering by 1d trend to avoid counter-trend trades.
# Timeframe: 6h, HTF: 1d as per experiment guidelines.

name = "6h_ElderRay_1dEMA50_Trend_VolumeConfirmation_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA13 for Elder Ray (on 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = Close - EMA13, Bear Power = Low - EMA13
    bull_power = close - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA13 and 1d EMA50
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_low = low[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0, Bear Power < 0, 1d uptrend, volume confirmation
            if (curr_bull_power > 0 and 
                curr_bear_power < 0 and 
                curr_close > curr_ema_50_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0, Bull Power < 0, 1d downtrend, volume confirmation
            elif (curr_bear_power < 0 and 
                  curr_bull_power < 0 and 
                  curr_close < curr_ema_50_1d and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: Bull Power <= 0 (bullish momentum fading)
            if curr_bull_power <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: Bear Power >= 0 (bearish momentum fading)
            if curr_bear_power >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals