#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Bull/Bear Power with 1d EMA34 trend filter and volume spike (>2x average)
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 and increasing + price above 1d EMA34 + volume spike
# Short when Bear Power < 0 and decreasing + price below 1d EMA34 + volume spike
# Uses 6h timeframe to target 50-150 total trades over 4 years (12-37/year)
# Volume filter >2x 20-period average reduces false signals
# Discrete position sizing: 0.25 for entries to limit fee drag
# Works in bull/bear regimes: trend filter aligns with higher timeframe, Elder Ray measures momentum

name = "6h_ElderRay_BullBearPower_1dEMA34_Volume_v1"
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
    
    # Calculate EMA13 for Elder Ray (6h)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power: High - EMA13
    bear_power = low - ema_13   # Bear Power: Low - EMA13
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 13, 20, 34)  # warmup for EMA13 (13), volume MA (20), EMA34 (34)
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_prev_bull_power = bull_power[i-1]
        curr_prev_bear_power = bear_power[i-1]
        curr_volume_spike = volume_spike[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with trend filter and Elder Ray momentum
            if curr_volume_spike:
                # Bullish: Bull Power > 0 and increasing + price above 1d EMA34
                if curr_bull_power > 0 and curr_bull_power > curr_prev_bull_power and curr_close > curr_ema_34_1d:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Bear Power < 0 and decreasing + price below 1d EMA34
                elif curr_bear_power < 0 and curr_bear_power < curr_prev_bear_power and curr_close < curr_ema_34_1d:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: Bull Power turns negative (momentum loss) or price below 1d EMA34 (trend break)
            if curr_bull_power <= 0 or curr_close < curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bear Power turns positive (momentum loss) or price above 1d EMA34 (trend break)
            if curr_bear_power >= 0 or curr_close > curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals