#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA200 trend filter and volume confirmation.
# Uses 4h for signal direction (price > EMA200 = long bias, price < EMA200 = short bias) and structure (Camarilla levels from 1d).
# 1h timeframe only for entry timing precision to avoid overtrading.
# Long when price breaks above Camarilla R1 with 4h EMA200 uptrend and volume > 1.5x 24-bar average.
# Short when price breaks below Camarilla S1 with 4h EMA200 downtrend and volume confirmation.
# Discrete sizing 0.20. ATR-based stoploss (signal→0 when price moves against position by 2.0*ATR).
# Session filter: 08-20 UTC to reduce noise trades.
# Target: 60-150 total trades over 4 years (15-37/year) to balance edge and fee drag.

name = "1h_Camarilla_R1S1_4hEMA200_Trend_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for 08-20 UTC filter
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 4h data ONCE before loop for EMA200 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA200 for trend filter
    ema_200 = pd.Series(df_4h['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_4h, ema_200)
    
    # Load 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla pivot levels from previous day's OHLC
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Pivot point (PP)
    pp = (prev_high + prev_low + prev_close) / 3.0
    # Resistance and Support levels
    r1 = pp + (prev_high - prev_low) * 1.1 / 12.0
    s1 = pp - (prev_high - prev_low) * 1.1 / 12.0
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    start_idx = 200  # warmup for 4h EMA200
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        if (np.isnan(ema_200_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x 24-bar average
        vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values[i]
        if vol_ma <= 0:
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_ma * 1.5)
        
        # Camarilla breakout conditions (using current bar's levels)
        breakout_up = curr_close > r1_aligned[i]  # break above R1
        breakout_down = curr_close < s1_aligned[i]  # break below S1
        
        # 4h EMA200 trend filter: above = uptrend (long bias), below = downtrend (short bias)
        uptrend = close[i] > ema_200_aligned[i]
        downtrend = close[i] < ema_200_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Camarilla breakout up AND uptrend AND volume confirmation
            if (breakout_up and 
                uptrend and 
                volume_confirm):
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            # Short: Camarilla breakout down AND downtrend AND volume confirmation
            elif (breakout_down and 
                  downtrend and 
                  volume_confirm):
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price re-enters Camarilla range OR trend reverses
            elif (curr_close < r1_aligned[i] and curr_close > s1_aligned[i]) or \
                 not uptrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price re-enters Camarilla range OR trend reverses
            elif (curr_close < r1_aligned[i] and curr_close > s1_aligned[i]) or \
                 not downtrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
    
    return signals