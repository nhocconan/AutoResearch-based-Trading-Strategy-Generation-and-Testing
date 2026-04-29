#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation
# Camarilla R3/S3 levels act as strong support/resistance on 4h timeframe.
# Breakouts with volume (>2.0x 20-period average) and 12h EMA50 trend alignment capture momentum moves.
# Designed for ~20-50 trades/year to minimize fee drag while participating in established trends.
# Uses strict volume confirmation to reduce false breakouts and overtrading.
# Works in bull/bear via 12h EMA50 trend filter - only trades in direction of 12h momentum.

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values  # needed for Camarilla calculation
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR (14-period) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    start_idx = 20  # volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_open = open_price[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_ema50_12h = ema_50_12h_aligned[i]
        curr_atr = atr[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Calculate Camarilla levels for this 4h bar using previous bar's OHLC
        if i == 0:
            signals[i] = 0.0
            continue
        prev_close = close[i-1]
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_open = open_price[i-1]
        
        # Camarilla levels (based on previous bar's range)
        R4 = prev_close + (prev_high - prev_low) * 1.1 / 2
        R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
        S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
        S4 = prev_close - (prev_high - prev_low) * 1.1 / 2
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: stoploss hit or price breaks below S3 (failed breakout)
            if curr_close < entry_price - 2.0 * atr_at_entry or curr_close < S3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: stoploss hit or price breaks above R3 (failed breakout)
            if curr_close > entry_price + 2.0 * atr_at_entry or curr_close > R3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new breakout entries
            # Volume confirmation: current volume > 2.0x 20-period average
            vol_confirm = curr_volume > 2.0 * curr_vol_ma
            
            # Long breakout when price closes above R3 with 12h EMA50 uptrend and volume confirmation
            if curr_close > R3 and curr_close > curr_ema50_12h and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_at_entry = curr_atr
            # Short breakout when price closes below S3 with 12h EMA50 downtrend and volume confirmation
            elif curr_close < S3 and curr_close < curr_ema50_12h and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_at_entry = curr_atr
            else:
                signals[i] = 0.0
    
    return signals