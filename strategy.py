#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Bull/Bear Power with 1d EMA50 trend filter and volume confirmation
# Elder Ray measures bull/bear power as price relative to EMA: Bull Power = High - EMA, Bear Power = Low - EMA
# Long when Bull Power > 0 and rising, Bear Power < 0 and falling, with price > 1d EMA50 (uptrend)
# Short when Bear Power < 0 and falling, Bull Power > 0 and rising, with price < 1d EMA50 (downtrend)
# Volume confirmation (>1.5x 24-period average) ensures institutional participation
# Designed for 6h timeframe to capture swings with controlled trade frequency (~15-30 trades/year)
# Works in both bull and bear markets by following the 1d EMA50 trend

name = "6h_ElderRay_BullBearPower_1dEMA50_VolumeConfirm_v1"
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
    
    # Get 1d data for EMA50 trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA13 for Elder Ray (same as used in Bull/Bear Power)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
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
    
    start_idx = max(50, 24, 13, 14)  # EMA50_1d, volume MA, EMA13, ATR warmup
    
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
        
        # Calculate previous bar values for momentum
        prev_bull_power = bull_power[i-1]
        prev_bear_power = bear_power[i-1]
        
        # Handle stoploss and exits
        if position == 1:  # Long position
            # Stoploss: price closes below entry - 2.0 * ATR_at_entry
            if curr_close < entry_price - 2.0 * atr_at_entry:
                signals[i] = 0.0
                position = 0
            # Exit: Bear Power turns positive (momentum loss) or price breaks below 1d EMA50
            elif curr_bear_power > 0 or curr_close < curr_ema50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Stoploss: price closes above entry + 2.0 * ATR_at_entry
            if curr_close > entry_price + 2.0 * atr_at_entry:
                signals[i] = 0.0
                position = 0
            # Exit: Bull Power turns negative (momentum loss) or price breaks above 1d EMA50
            elif curr_bull_power < 0 or curr_close > curr_ema50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 24-period average
            vol_confirm = curr_volume > 1.5 * curr_vol_ma
            
            # Long entry: Bull Power > 0 and rising, Bear Power < 0, price > 1d EMA50 (uptrend)
            if vol_confirm and curr_close > curr_ema50_1d:
                if curr_bull_power > 0 and curr_bull_power > prev_bull_power and curr_bear_power < 0:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
            # Short entry: Bear Power < 0 and falling, Bull Power > 0, price < 1d EMA50 (downtrend)
            elif vol_confirm and curr_close < curr_ema50_1d:
                if curr_bear_power < 0 and curr_bear_power < prev_bear_power and curr_bull_power > 0:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
            else:
                signals[i] = 0.0
    
    return signals