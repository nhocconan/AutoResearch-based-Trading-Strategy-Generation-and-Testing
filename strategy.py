#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike confirmation
# Uses 4h HTF for trend direction (more stable than 1h) and 1h for precise entry timing
# Session filter (08-20 UTC) reduces noise trades during low-volume periods
# Discrete position sizing: 0.20 to minimize fee churn
# Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeSpike_Session_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla pivot levels from previous day
        # Need at least 24 hours of data (24 1h bars) for previous day
        if i >= 24:
            # Get previous day's high, low, close (24*1h bars = 24h)
            prev_day_high = np.max(high[i-24:i])   # Previous day's high (excluding current bar)
            prev_day_low = np.min(low[i-24:i])     # Previous day's low
            prev_day_close = close[i-1]            # Previous day's close (1 bar ago)
            
            # Calculate Camarilla levels
            range_val = prev_day_high - prev_day_low
            camarilla_h3 = prev_day_close + range_val * 1.1 / 6
            camarilla_l3 = prev_day_close - range_val * 1.1 / 6
            camarilla_h3l3_mid = (camarilla_h3 + camarilla_l3) / 2.0
        else:
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_4h = ema_50_4h_aligned[i]
        curr_atr = atr[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 2.0 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Stoploss: 2 * ATR below entry
            stop_price = entry_price - 2.0 * curr_atr
            # Exit conditions: Close below H3/L3 midpoint OR price below 4h EMA50 OR stoploss hit
            if curr_close < camarilla_h3l3_mid or curr_close < curr_ema_4h or curr_close < stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry
            stop_price = entry_price + 2.0 * curr_atr
            # Exit conditions: Close above H3/L3 midpoint OR price above 4h EMA50 OR stoploss hit
            if curr_close > camarilla_h3l3_mid or curr_close > curr_ema_4h or curr_close > stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long entry: Close > Camarilla H3 AND price > 4h EMA50 AND volume spike
            if (curr_close > camarilla_h3 and 
                curr_close > curr_ema_4h and
                vol_spike):
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            # Short entry: Close < Camarilla L3 AND price < 4h EMA50 AND volume spike
            elif (curr_close < camarilla_l3 and 
                  curr_close < curr_ema_4h and
                  vol_spike):
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals