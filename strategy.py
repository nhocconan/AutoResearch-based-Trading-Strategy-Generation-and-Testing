#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R Extreme with 1w EMA34 trend filter and volume confirmation.
# Uses weekly EMA34 for trend direction and requires volume > 2.0x average for confirmation.
# Williams %R < -80 for long entry, > -20 for short entry, only in direction of weekly trend.
# Designed for very low trade frequency (~30-80 total trades over 4 years) to minimize fee drag.
# Works in bull/bear markets by only taking extreme mean-reversion entries in the direction of the weekly trend.
# Added ATR-based stoploss (2.5x ATR) to manage risk and reduce whipsaws.

name = "1d_WilliamsR_Extreme_1wEMA34_VolumeConfirm_v2"
timeframe = "1d"
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
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate ATR for stoploss (using 14-period ATR on 1d)
    if n >= 14:
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    else:
        atr = np.full(n, np.nan)
    
    # Calculate Williams %R (14-period)
    if n >= 14:
        highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
        lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        # Handle division by zero (when high == low)
        williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    else:
        williams_r = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 34  # warmup for weekly EMA34 and Williams %R
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(williams_r[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_34_1w = ema_34_1w_aligned[i]
        curr_atr = atr[i]
        curr_williams_r = williams_r[i]
        
        # Volume confirmation: volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > (2.0 * vol_ma_20)
        else:
            volume_confirm = False
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R < -80 (oversold), weekly EMA34 uptrend, volume spike
            if (curr_williams_r < -80 and 
                curr_close > curr_ema_34_1w and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Williams %R > -20 (overbought), weekly EMA34 downtrend, volume spike
            elif (curr_williams_r > -20 and 
                  curr_close < curr_ema_34_1w and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit conditions: Williams %R > -50 (exit extreme), or ATR stoploss hit
            if curr_williams_r > -50 or curr_close < entry_price - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: Williams %R < -50 (exit extreme), or ATR stoploss hit
            if curr_williams_r < -50 or curr_close > entry_price + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals