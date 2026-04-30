#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Uses tight volume threshold (2.0x average) and ATR(14) stoploss (2.0x) to limit trades to ~150 total over 4 years.
# Only enters when price breaks 4h Camarilla R3 (short) or S3 (long) levels with volume confirmation and 1d EMA34 trend alignment.
# Designed for low trade frequency (<200 total 4h trades) to avoid fee drag. Works in bull/bear via 1d EMA34 trend filter.

name = "4h_Camarilla_R3S3_1dEMA34_VolumeConfirm_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
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
    
    start_idx = max(1, 34, atr_period, 20) + 1  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_atr = atr[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Calculate Camarilla levels from previous 1d bar
        # Use completed 1d bar: i-1 index in 1d data aligned to current 4h bar
        if i >= 1:
            # Get the previous completed 1d bar's OHLC from aligned arrays
            # Since we're on 4h timeframe, we need to get the 1d values from the previous day
            # The aligned arrays already give us the 1d values for each 4h bar
            # But for Camarilla, we need the previous 1d bar's data
            # We'll use the 1d data shifted by 1 bar (previous completed day)
            if len(df_1d) >= 2:
                # Get index of previous completed 1d bar
                prev_1d_idx = len(df_1d) - 2  # second to last completed 1d bar
                if prev_1d_idx >= 0:
                    ph = df_1d['high'].iloc[prev_1d_idx]
                    pl = df_1d['low'].iloc[prev_1d_idx]
                    pc = df_1d['close'].iloc[prev_1d_idx]
                    
                    # Calculate Camarilla levels
                    rang = ph - pl
                    r3 = pc + (rang * 1.1 / 4)
                    s3 = pc - (rang * 1.1 / 4)
                else:
                    r3 = np.nan
                    s3 = np.nan
            else:
                r3 = np.nan
                s3 = np.nan
        else:
            r3 = np.nan
            s3 = np.nan
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above S3, 1d EMA34 uptrend, volume spike confirmation
            if (curr_close > s3 and 
                curr_close > curr_ema_34_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
            # Short: price breaks below R3, 1d EMA34 downtrend, volume spike confirmation
            elif (curr_close < r3 and 
                  curr_close < curr_ema_34_1d and 
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