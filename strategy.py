#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike confirmation
# Long: Close > Camarilla R3 AND price > 4h EMA50 AND volume > 2.0x 20-bar avg
# Short: Close < Camarilla S3 AND price < 4h EMA50 AND volume > 2.0x 20-bar avg
# Exit: Close crosses Camarilla H3/L3 OR price crosses 4h EMA50 OR ATR stoploss
# ATR stoploss: 2.0 * ATR(14) from entry price
# Uses 4h for signal direction, 1h for entry timing to reduce noise
# Session filter: 08-20 UTC to avoid low-volume hours
# Discrete position sizing: 0.20 for long/short, 0.0 for flat to minimize fee churn
# Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeSpike_v1"
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
    
    start_idx = max(50, 20, 14)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        if hours[i] < 8 or hours[i] > 20:
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels (using previous bar's OHLC)
        if i >= 1:
            prev_close = close[i-1]
            prev_high = high[i-1]
            prev_low = low[i-1]
            cam_range = prev_high - prev_low
            camarilla_h3 = prev_close + cam_range * 1.1 / 4
            camarilla_l3 = prev_close - cam_range * 1.1 / 4
            camarilla_r3 = prev_close + cam_range * 1.1 / 2
            camarilla_s3 = prev_close - cam_range * 1.1 / 2
            camarilla_h4 = prev_close + cam_range * 1.1 / 2
            camarilla_s4 = prev_close - cam_range * 1.1 / 2
            camarilla_h5 = prev_close + cam_range * 1.1 * 2
            camarilla_s5 = prev_close - cam_range * 1.1 * 2
            camarilla_h6 = prev_close + cam_range * 1.1 * 2.5
            camarilla_s6 = prev_close - cam_range * 1.1 * 2.5
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
            # Exit conditions: Close below Camarilla H3/L3 OR price below 4h EMA50 OR stoploss hit
            if curr_close < camarilla_h3 or curr_close < camarilla_l3 or curr_close < curr_ema_4h or curr_close < stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry
            stop_price = entry_price + 2.0 * curr_atr
            # Exit conditions: Close above Camarilla H3/L3 OR price above 4h EMA50 OR stoploss hit
            if curr_close > camarilla_h3 or curr_close > camarilla_l3 or curr_close > curr_ema_4h or curr_close > stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long entry: Close > Camarilla R3 AND price > 4h EMA50 AND volume spike
            if (curr_close > camarilla_r3 and 
                curr_close > curr_ema_4h and
                vol_spike):
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            # Short entry: Close < Camarilla S3 AND price < 4h EMA50 AND volume spike
            elif (curr_close < camarilla_s3 and 
                  curr_close < curr_ema_4h and
                  vol_spike):
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals