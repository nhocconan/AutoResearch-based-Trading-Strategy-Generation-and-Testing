#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Bull/Bear Power with 1d EMA50 trend filter and volume spike
# Uses Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) to measure bull/bear strength
# Trend filter: 1d EMA50 to ensure trades align with higher timeframe direction
# Volume confirmation: >2.0x 24-period average to filter low-quality breakouts
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag
# Designed for 6h timeframe to capture medium-term swings with controlled frequency
# Works in both bull and bear markets by using trend-filtered Elder Ray extremes

name = "6h_ElderRay_BullBearPower_1dEMA50_VolumeSpike_v2"
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA13 for Elder Ray (using primary timeframe)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull Power: High - EMA13
    bear_power = low - ema_13   # Bear Power: Low - EMA13
    
    # Calculate 24-period average volume for confirmation
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    # Calculate ATR(14) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    start_idx = max(50, 24, 13, 14)  # EMA50, volume MA, EMA13, and ATR warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_atr = atr[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_24[i]
        
        # Handle stoploss and exits
        if position == 1:  # Long position
            # Stoploss: price closes below entry - 2.0 * ATR_at_entry
            if curr_close < entry_price - 2.0 * atr_at_entry:
                signals[i] = 0.0
                position = 0
            # Exit: Bear Power turns negative or trend turns down
            elif curr_bear_power >= 0 or curr_close < curr_ema50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Stoploss: price closes above entry + 2.0 * ATR_at_entry
            if curr_close > entry_price + 2.0 * atr_at_entry:
                signals[i] = 0.0
                position = 0
            # Exit: Bull Power turns positive or trend turns up
            elif curr_bull_power <= 0 or curr_close > curr_ema50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 24-period average
            vol_confirm = curr_volume > 2.0 * curr_vol_ma
            
            # Long entry: Bull Power > 0 (strong bullishness) in uptrend (price > EMA50_1d)
            if vol_confirm and curr_close > curr_ema50_1d:
                if curr_bull_power > 0:  # Bullish momentum
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
            # Short entry: Bear Power < 0 (strong bearishness) in downtrend (price < EMA50_1d)
            elif vol_confirm and curr_close < curr_ema50_1d:
                if curr_bear_power < 0:  # Bearish momentum
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
            else:
                signals[i] = 0.0
    
    return signals