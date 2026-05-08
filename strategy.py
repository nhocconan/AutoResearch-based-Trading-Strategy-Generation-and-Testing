#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power (Bull/Bear) with 1d Trend Filter and Volume Spike
# - Elder Ray: Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# - Long when Bull Power > 0 and rising, Bear Power < 0 and falling, with 1d uptrend
# - Short when Bear Power > 0 and rising, Bull Power < 0 and falling, with 1d downtrend
# - Volume spike filter to avoid low-conviction moves
# - Works in bull/bear by using 1d trend to align with higher timeframe momentum

name = "6h_ElderRay_Power_1dTrend_Volume"
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
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Elder Ray components: EMA13 of close
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Higher highs vs trend
    bear_power = ema_13 - low   # Lower lows vs trend
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power rising (momentum building) + Bear Power falling + 1d uptrend + volume spike
            bull_rising = bull_power[i] > bull_power[i-1]
            bear_falling = bear_power[i] < bear_power[i-1]
            long_cond = bull_rising and bear_falling and (ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]) and volume_spike[i]
            
            # Short: Bear Power rising (momentum building down) + Bull Power falling + 1d downtrend + volume spike
            bear_rising = bear_power[i] > bear_power[i-1]
            bull_falling = bull_power[i] < bull_power[i-1]
            short_cond = bear_rising and bull_falling and (ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]) and volume_spike[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bear Power rising (sellers taking over) or loss of 1d uptrend
            bear_rising = bear_power[i] > bear_power[i-1]
            trend_lost = ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]
            if bear_rising or trend_lost:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bull Power rising (buyers taking over) or loss of 1d downtrend
            bull_rising = bull_power[i] > bull_power[i-1]
            trend_lost = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            if bull_rising or trend_lost:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals