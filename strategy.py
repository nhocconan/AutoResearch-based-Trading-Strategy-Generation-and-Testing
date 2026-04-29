#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w Supertrend trend filter and volume confirmation
# Uses 1w Supertrend for robust trend detection in both bull/bear markets, Camarilla levels for precise
# entry/exit at key support/resistance, and volume spike to confirm breakout strength.
# ATR-based stoploss manages risk. Designed for low trade frequency (target: 15-25/year) to minimize fee drag.

name = "1d_Camarilla_R3S3_Breakout_1wSupertrend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Load HTF data ONCE before loop for 1w calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w Supertrend (ATR=10, mult=3.0)
    hl2_1w = (df_1w['high'] + df_1w['low']) / 2
    atr_1w = pd.Series(
        np.maximum(
            np.maximum(df_1w['high'] - df_1w['low'],
                       np.abs(df_1w['high'] - df_1w['close'].shift(1))),
            np.abs(df_1w['low'] - df_1w['close'].shift(1))
        )
    ).rolling(window=10, min_periods=10).mean()
    
    upper_band_1w = hl2_1w + (3.0 * atr_1w)
    lower_band_1w = hl2_1w - (3.0 * atr_1w)
    
    supertrend_1w = np.full(len(df_1w), np.nan, dtype=float)
    direction_1w = np.full(len(df_1w), np.nan, dtype=float)  # 1 for uptrend, -1 for downtrend
    
    for i in range(10, len(df_1w)):
        if i == 10:
            supertrend_1w[i] = upper_band_1w.iloc[i]
            direction_1w[i] = 1
        else:
            if supertrend_1w[i-1] == upper_band_1w.iloc[i-1]:
                supertrend_1w[i] = upper_band_1w.iloc[i] if close_1w.iloc[i] <= upper_band_1w.iloc[i] else lower_band_1w.iloc[i]
                direction_1w[i] = -1 if supertrend_1w[i] == upper_band_1w.iloc[i] else 1
            else:
                supertrend_1w[i] = lower_band_1w.iloc[i] if close_1w.iloc[i] >= lower_band_1w.iloc[i] else upper_band_1w.iloc[i]
                direction_1w[i] = 1 if supertrend_1w[i] == lower_band_1w.iloc[i] else -1
    
    # Extract close_1w for comparison
    close_1w = df_1w['close']
    
    # Supertrend values and direction
    supertrend_1w_vals = supertrend_1w.values
    supertrend_dir_1w = direction_1w.values
    
    # Align to 1d timeframe
    supertrend_1w_aligned = align_htf_to_ltf(prices, df_1w, supertrend_1w_vals)
    supertrend_dir_1w_aligned = align_htf_to_ltf(prices, df_1w, supertrend_dir_1w)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla levels from previous day
    prev_close = np.concatenate([[np.nan], close[:-1]])
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    
    camarilla_r3 = prev_close + 1.0 * (prev_high - prev_low)
    camarilla_s3 = prev_close - 1.0 * (prev_high - prev_low)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    max_high_since_entry = 0.0
    min_low_since_entry = 0.0
    
    start_idx = max(50, 20, 14)  # warmup for Supertrend, volume, ATR
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(supertrend_1w_aligned[i]) or np.isnan(supertrend_dir_1w_aligned[i]) or \
           np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_open = open_price[i]
        curr_supertrend = supertrend_1w_aligned[i]
        curr_supertrend_dir = supertrend_dir_1w_aligned[i]
        curr_atr = atr[i]
        curr_r3 = camarilla_r3[i]
        curr_s3 = camarilla_s3[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Handle position exits and stops
        if position == 1:  # Long position
            # Update trailing stop: highest high since entry
            max_high_since_entry = max(max_high_since_entry, curr_high)
            # Dynamic stoploss: ATR-based trailing stop
            trail_stop = max_high_since_entry - 2.5 * curr_atr
            # Fixed stoploss: 2.0 * ATR below entry
            fixed_stop = entry_price - 2.0 * atr_at_entry
            # Use the tighter of the two stops
            stop_price = max(trail_stop, fixed_stop)
            
            # Exit conditions:
            # 1. Stoploss hit (trailing or fixed)
            # 2. Supertrend turns bearish (trend change)
            # 3. Price drops below Camarilla S3 (breakout failed)
            if (curr_low <= stop_price or
                curr_supertrend_dir == -1 or
                curr_close < curr_s3):
                signals[i] = 0.0
                position = 0
                max_high_since_entry = 0.0
                min_low_since_entry = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update trailing stop: lowest low since entry
            min_low_since_entry = min(min_low_since_entry, curr_low)
            # Dynamic stoploss: ATR-based trailing stop
            trail_stop = min_low_since_entry + 2.5 * curr_atr
            # Fixed stoploss: 2.0 * ATR above entry
            fixed_stop = entry_price + 2.0 * atr_at_entry
            # Use the tighter of the two stops
            stop_price = min(trail_stop, fixed_stop)
            
            # Exit conditions:
            # 1. Stoploss hit (trailing or fixed)
            # 2. Supertrend turns bullish (trend change)
            # 3. Price rises above Camarilla R3 (breakout failed)
            if (curr_high >= stop_price or
                curr_supertrend_dir == 1 or
                curr_close > curr_r3):
                signals[i] = 0.0
                position = 0
                max_high_since_entry = 0.0
                min_low_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 + Supertrend bullish + volume confirm
            if (curr_close > curr_r3 and
                curr_supertrend_dir == 1 and
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_at_entry = curr_atr
                max_high_since_entry = curr_high
                min_low_since_entry = curr_low
            # Short entry: price breaks below Camarilla S3 + Supertrend bearish + volume confirm
            elif (curr_close < curr_s3 and
                  curr_supertrend_dir == -1 and
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_at_entry = curr_atr
                max_high_since_entry = curr_high
                min_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals