#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA34 trend filter and volume confirmation.
# Uses Donchian channel from prior 20 periods for structure, 12h EMA34 for trend alignment,
# volume > 2.0x 20-bar average for confirmation, and ATR(14) trailing stop (2.5x) for risk management.
# Discrete position sizing at ±0.30 to balance return and drawdown. Target: 100-180 total trades over 4 years (25-45/year).
# Session filter (08:00-20:00 UTC) to avoid low-liquidity periods.

name = "4h_Donchian20_12hEMA34_VolumeSpike_ATRStop_v1"
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
    
    # Load 12h data ONCE before loop for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend filter
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Donchian(20) from prior 20 periods (use shift(1) to avoid look-ahead)
    donchian_period = 20
    highest_20 = pd.Series(high).shift(1).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_20 = pd.Series(low).shift(1).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # ATR(14) for volatility and stoploss
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(50, donchian_period, atr_period, 20) + 1  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_34_12h_aligned[i]) or
            np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i]) or
            np.isnan(atr[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_34_12h = ema_34_12h_aligned[i]
        curr_upper = highest_20[i]
        curr_lower = lowest_20[i]
        curr_atr = atr[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper, above 12h EMA34, volume confirmation
            if (curr_close > curr_upper and 
                curr_close > curr_ema_34_12h and 
                curr_volume_confirm):
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
            # Short: price breaks below Donchian lower, below 12h EMA34, volume confirmation
            elif (curr_close < curr_lower and 
                  curr_close < curr_ema_34_12h and 
                  curr_volume_confirm):
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
                lowest_since_entry = curr_close
        
        elif position == 1:  # Long position
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # ATR trailing stop: exit if price drops 2.5*ATR from highest point
            if curr_close < highest_since_entry - (2.5 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # ATR trailing stop: exit if price rises 2.5*ATR from lowest point
            if curr_close > lowest_since_entry + (2.5 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals