#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter, volume spike, and chop regime filter.
# Uses tighter Camarilla levels (R3/S3) for precise entries. Adds choppiness index (CHOP) > 61.8 to avoid trending markets,
# focusing on mean-reversion breakouts in ranging conditions. Volume > 2.0x 20-bar average for confirmation.
# ATR(14) trailing stop at 2.0x for tighter risk control. Discrete position sizing at ±0.25.
# Target: 50-150 total trades over 4 years (12-38/year) to avoid fee drag. Works in both bull and bear via regime filter.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_CHOP_v1"
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
    
    # Load 1d data ONCE before loop for EMA34 trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from prior 1d OHLC
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    camarilla_r3 = daily_close + 1.1 * (daily_high - daily_low) * 1.1 / 4
    camarilla_s3 = daily_close - 1.1 * (daily_high - daily_low) * 1.1 / 4
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # ATR(14) for volatility and stoploss
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume confirmation: volume > 2.0x 20-period average (tighter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    # Choppiness Index (CHOP) regime filter: CHOP > 61.8 = ranging (favor mean-reversion breakouts)
    chop_period = 14
    tr_sum = pd.Series(tr).rolling(window=chop_period, min_periods=chop_period).sum().values
    highest_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    chop = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(chop_period)
    chop_filter = chop > 61.8  # Only trade in ranging markets
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(50, 34, atr_period, 20, chop_period) + 1  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session or not ranging
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(volume_confirm[i]) or
            np.isnan(chop[i]) or
            not in_session[i] or
            not chop_filter[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_atr = atr[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3, above 1d EMA34, volume confirmation, in ranging market
            if (curr_close > curr_r3 and 
                curr_close > curr_ema_34_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
            # Short: price breaks below Camarilla S3, below 1d EMA34, volume confirmation, in ranging market
            elif (curr_close < curr_s3 and 
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