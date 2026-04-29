#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 AND increasing AND price > 1d EMA34 (bullish regime)
# Short when Bear Power < 0 AND decreasing AND price < 1d EMA34 (bearish regime)
# Volume spike confirms momentum. Works in both bull/bear markets by following 1d trend.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "6h_ElderRay_1dEMA34_Trend_VolumeSpike_v1"
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
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 6h timeframe (completed 1d bar only)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate EMA(13) for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power: High - EMA13
    bear_power = low - ema_13   # Bear Power: Low - EMA13
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34, 20, 13)  # warmup for EMA34, volume MA, and EMA13
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema34_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_ema34 = ema34_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Trend regime: bullish if price > 1d EMA34, bearish if price < 1d EMA34
        is_bullish_regime = curr_close > curr_ema34
        is_bearish_regime = curr_close < curr_ema34
        
        # Elder Ray momentum: Bull Power increasing, Bear Power decreasing
        bull_power_increasing = i > start_idx and curr_bull_power > bull_power[i-1]
        bear_power_decreasing = i > start_idx and curr_bear_power < bear_power[i-1]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Bullish entry: Bull Power > 0 AND increasing AND bullish regime
                if curr_bull_power > 0 and bull_power_increasing and is_bullish_regime:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Bear Power < 0 AND decreasing AND bearish regime
                elif curr_bear_power < 0 and bear_power_decreasing and is_bearish_regime:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: Bull Power <= 0 OR regime changes to bearish OR Bull Power stops increasing
            if curr_bull_power <= 0 or not is_bullish_regime or not bull_power_increasing:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: Bear Power >= 0 OR regime changes to bullish OR Bear Power stops decreasing
            if curr_bear_power >= 0 or not is_bearish_regime or not bear_power_decreasing:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals