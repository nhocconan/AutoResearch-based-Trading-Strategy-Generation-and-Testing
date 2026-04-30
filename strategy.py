#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation.
# Uses volume > 2.0x 20-period average and ATR(14) stoploss (2.0x) to limit trades.
# Designed for low trade frequency (<200 total 4h trades) to avoid fee drag.
# Works in bull/bear via 12h EMA50 trend filter and session restriction (08-20 UTC).

name = "4h_Camarilla_R3S3_12hEMA50_VolumeConfirm_ATRStop_v1"
timeframe = "4h"
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
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # ATR(14) for volatility and stoploss
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume confirmation: volume > 2.0x 20-period average (tight threshold to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(1, 50, atr_period, 20) + 1  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_atr = atr[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Calculate Camarilla pivot levels from previous day (using 1d data)
        # We need to get the 1d data for pivot calculation
        if i >= 6:  # Need at least one 1d bar (6x4h) to calculate
            # For simplicity, we'll use a lookback of 6 periods (1d = 6x4h)
            if i >= 6:
                # Get high, low, close of previous 1d bar (6 periods ago)
                prev_day_high = np.max(high[i-6:i])
                prev_day_low = np.min(low[i-6:i])
                prev_day_close = close[i-1]  # approximate previous close
            else:
                prev_day_high = np.nan
                prev_day_low = np.nan
                prev_day_close = np.nan
        else:
            prev_day_high = np.nan
            prev_day_low = np.nan
            prev_day_close = np.nan
        
        # Calculate Camarilla R3, S3 levels
        if not (np.isnan(prev_day_high) or np.isnan(prev_day_low) or np.isnan(prev_day_close)):
            pivot = (prev_day_high + prev_day_low + prev_day_close) / 3
            range_ = prev_day_high - prev_day_low
            r3 = pivot + (range_ * 1.1 / 4)
            s3 = pivot - (range_ * 1.1 / 4)
        else:
            r3 = np.nan
            s3 = np.nan
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3, 12h EMA50 uptrend, volume spike confirmation
            if (curr_close > r3 and 
                curr_close > curr_ema_50_12h and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
            # Short: price breaks below Camarilla S3, 12h EMA50 downtrend, volume spike confirmation
            elif (curr_close < s3 and 
                  curr_close < curr_ema_50_12h and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                lowest_since_entry = curr_close
        
        elif position == 1:  # Long position
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # ATR trailing stop: exit if price drops 2.0*ATR from highest point
            if curr_close < highest_since_entry - (2.0 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # ATR trailing stop: exit if price rises 2.0*ATR from lowest point
            if curr_close > lowest_since_entry + (2.0 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals