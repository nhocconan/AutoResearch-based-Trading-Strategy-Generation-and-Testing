#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w Supertrend filter and volume confirmation
# Camarilla levels provide institutional support/resistance; 1w Supertrend filters for higher timeframe trend direction;
# volume confirms breakout strength; fixed ATR-based stoploss manages risk.
# Designed to work in both bull and bear markets by only taking trades in direction of weekly trend.
# Target: 20-30 trades/year (80-120 total over 4 years) to balance opportunity with fee drag.

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
    
    # Calculate 1w Supertrend for trend filter
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = df_1w['high'][1:] - df_1w['low'][1:]
    tr2 = np.abs(df_1w['high'][1:] - df_1w['close'][:-1])
    tr3 = np.abs(df_1w['low'][1:] - df_1w['close'][:-1])
    tr_1w = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1w = pd.Series(tr_1w).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (df_1w['high'] + df_1w['low']) / 2
    basic_ub = hl2 + multiplier * atr_1w
    basic_lb = hl2 - multiplier * atr_1w
    
    # Final Upper and Lower Bands
    final_ub = np.zeros_like(basic_ub)
    final_lb = np.zeros_like(basic_lb)
    final_ub[0] = basic_ub[0]
    final_lb[0] = basic_lb[0]
    
    for i in range(1, len(basic_ub)):
        if basic_ub[i] < final_ub[i-1] or df_1w['close'].iloc[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i-1]
            
        if basic_lb[i] > final_lb[i-1] or df_1w['close'].iloc[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i-1]
    
    # Supertrend
    supertrend = np.zeros_like(final_ub)
    supertrend[0] = final_ub[0]
    
    for i in range(1, len(final_ub)):
        if supertrend[i-1] == final_ub[i-1]:
            if df_1w['close'].iloc[i] <= final_ub[i]:
                supertrend[i] = final_ub[i]
            else:
                supertrend[i] = final_lb[i]
        else:
            if df_1w['close'].iloc[i] >= final_lb[i]:
                supertrend[i] = final_lb[i]
            else:
                supertrend[i] = final_ub[i]
    
    # Supertrend trend direction: 1 = uptrend, -1 = downtrend
    supertrend_direction = np.where(close_1w := df_1w['close'].values > supertrend, 1, -1)
    supertrend_direction_aligned = align_htf_to_ltf(prices, df_1w, supertrend_direction)
    
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
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = prev_close + 1.0 * (prev_high - prev_low)
    camarilla_s3 = prev_close - 1.0 * (prev_high - prev_low)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    start_idx = max(50, 20, 14)  # warmup for Supertrend, volume, ATR
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(supertrend_direction_aligned[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_open = open_price[i]
        curr_supertrend_dir = supertrend_direction_aligned[i]
        curr_atr = atr[i]
        curr_r3 = camarilla_r3[i]
        curr_s3 = camarilla_s3[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Handle position exits and stops
        if position == 1:  # Long position
            # Fixed stoploss: 2.5 * ATR below entry
            stop_price = entry_price - 2.5 * atr_at_entry
            
            # Exit conditions:
            # 1. Stoploss hit
            # 2. Weekly trend turns bearish
            # 3. Price drops below Camarilla S3 (breakout failed)
            if (curr_low <= stop_price or
                curr_supertrend_dir == -1 or
                curr_close < curr_s3):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Fixed stoploss: 2.5 * ATR above entry
            stop_price = entry_price + 2.5 * atr_at_entry
            
            # Exit conditions:
            # 1. Stoploss hit
            # 2. Weekly trend turns bullish
            # 3. Price rises above Camarilla R3 (breakout failed)
            if (curr_high >= stop_price or
                curr_supertrend_dir == 1 or
                curr_close > curr_r3):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Only take longs in weekly uptrend, shorts in weekly downtrend
            if curr_supertrend_dir == 1:  # Weekly uptrend - look for longs
                if (curr_close > curr_r3 and
                    curr_volume_confirm):
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
            elif curr_supertrend_dir == -1:  # Weekly downtrend - look for shorts
                if (curr_close < curr_s3 and
                    curr_volume_confirm):
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
            else:
                signals[i] = 0.0
    
    return signals