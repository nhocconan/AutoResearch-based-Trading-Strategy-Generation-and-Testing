#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA50 trend filter and volume confirmation.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low (using 13-period EMA as per Elder).
# Long when Bull Power > 0 AND Bear Power < 0 AND price > 12h EMA50 AND volume > 1.5x 20-bar avg.
# Short when Bear Power > 0 AND Bull Power < 0 AND price < 12h EMA50 AND volume > 1.5x 20-bar avg.
# Exit when power signals reverse (Bull Power <= 0 for long exit, Bear Power <= 0 for short exit).
# Uses 12h EMA50 as trend filter to avoid counter-trend trades, volume confirmation to reduce false signals.
# Elder Ray captures momentum via power metrics, effective in both trending and ranging markets.
# Timeframe: 6h, HTF: 12h as per experiment guidelines.

name = "6h_ElderRay_12hEMA50_Trend_VolumeSpike_v1"
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
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Elder Ray components using 13-period EMA of close
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = ema_13 - low   # Bear Power = EMA13 - Low
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 and EMA13
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0, Bear Power < 0, uptrend (close > 12h EMA50), volume spike
            if (curr_bull_power > 0 and 
                curr_bear_power < 0 and 
                curr_close > curr_ema_50_12h and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0, Bull Power < 0, downtrend (close < 12h EMA50), volume spike
            elif (curr_bear_power > 0 and 
                  curr_bull_power < 0 and 
                  curr_close < curr_ema_50_12h and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: Bull Power <= 0 (momentum weakening)
            if curr_bull_power <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: Bear Power <= 0 (momentum weakening)
            if curr_bear_power <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals