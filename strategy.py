#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Supertrend for trend direction, 1h only for entry timing precision.
# Uses 4h Supertrend (ATR=10, mult=3.0) to filter trend direction and avoid counter-trend trades.
# Enters on 1h pullbacks to EMA21 in direction of 4h trend with volume confirmation (>1.5x 20-bar avg).
# Exits on opposite Supertrend signal or ATR(14) trailing stop (2.0x). Discrete sizing ±0.20 to limit fee drag.
# Session filter (08:00-20:00 UTC) to reduce noise. Target: 80-120 total trades over 4 years (20-30/year).
# Designed to work in both bull (trend following) and bear (avoids counter-trend, captures retracements).

name = "1h_Supertrend4h_EMA21_Pullback_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for Supertrend trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h ATR(10) for Supertrend
    atr_period = 10
    tr1 = df_4h['high'][1:] - df_4h['low'][1:]
    tr2 = np.abs(df_4h['high'][1:] - df_4h['close'][:-1])
    tr3 = np.abs(df_4h['low'][1:] - df_4h['close'][:-1])
    tr_4h = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_4h = pd.Series(tr_4h).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Calculate 4h Supertrend
    hl2_4h = (df_4h['high'] + df_4h['low']) / 2
    upper_band = hl2_4h + (3.0 * atr_4h)
    lower_band = hl2_4h - (3.0 * atr_4h)
    
    # Initialize Supertrend arrays
    supertrend = np.zeros_like(hl2_4h)
    direction = np.ones_like(hl2_4h)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = hl2_4h[0]
    direction[0] = 1
    
    for i in range(1, len(hl2_4h)):
        if close_4h := df_4h['close'].iloc[i]:
            pass  # Just to reference close for type checking
        
        if df_4h['close'].iloc[i] > supertrend[i-1]:
            direction[i] = 1
        else:
            direction[i] = -1
        
        if direction[i] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    # Align 4h Supertrend and direction to 1h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_4h, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_4h, direction)
    
    # Calculate 1h EMA21 for pullback entries
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # ATR(14) for volatility and stoploss on 1h
    atr_period_1h = 14
    tr1_1h = high[1:] - low[1:]
    tr2_1h = np.abs(high[1:] - close[:-1])
    tr3_1h = np.abs(low[1:] - close[:-1])
    tr_1h = np.concatenate([[np.max([tr1_1h[0], tr2_1h[0], tr3_1h[0]])], np.maximum(tr1_1h, np.maximum(tr2_1h, tr3_1h))])
    atr_1h = pd.Series(tr_1h).rolling(window=atr_period_1h, min_periods=atr_period_1h).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = 21  # warmup for EMA21 and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(supertrend_aligned[i]) or
            np.isnan(direction_aligned[i]) or
            np.isnan(ema_21[i]) or
            np.isnan(atr_1h[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_supertrend = supertrend_aligned[i]
        curr_direction = direction_aligned[i]
        curr_ema_21 = ema_21[i]
        curr_atr = atr_1h[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: 4h uptrend, price pulls back to EMA21 or above, volume confirmation
            if (curr_direction == 1 and 
                curr_close >= curr_ema_21 and 
                curr_volume_confirm):
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
            # Short: 4h downtrend, price pulls back to EMA21 or below, volume confirmation
            elif (curr_direction == -1 and 
                  curr_close <= curr_ema_21 and 
                  curr_volume_confirm):
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
                lowest_since_entry = curr_close
        
        elif position == 1:  # Long position
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit conditions: 4h trend turns down OR ATR trailing stop (2.0x ATR)
            if (curr_direction == -1 or 
                curr_close < highest_since_entry - (2.0 * curr_atr)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit conditions: 4h trend turns up OR ATR trailing stop (2.0x ATR)
            if (curr_direction == 1 or 
                curr_close > lowest_since_entry + (2.0 * curr_atr)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals