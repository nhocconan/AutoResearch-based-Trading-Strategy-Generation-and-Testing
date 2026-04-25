#!/usr/bin/env python3
"""
4h Camarilla R3/S3 Breakout + 12h EMA50 Trend + Volume Spike
Hypothesis: Camarilla R3/S3 levels act as strong intraday support/resistance. Breakouts above R3 or below S3 with 12h EMA50 trend alignment capture momentum moves. Volume spike confirms institutional participation. Works in bull markets via buying R3 breakouts, bear markets via selling S3 breakdowns. Discrete position sizing (0.25) controls drawdown. Target: 25-40 trades/year on 4h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR(14) for stoploss
    if len(close) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean().values
    else:
        atr = np.full(n, 0.0)
    
    # Pre-compute 20-period volume MA for volume spike detection
    vol_ma_20 = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 19)
        vol_ma_20[i] = np.mean(volume[start_idx:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for ATR(14) and EMA50 to propagate
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_50 = ema_50_12h_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        
        # Calculate Camarilla pivot levels for today (using previous day's OHLC)
        # We need to get the previous day's high, low, close
        if i >= 96:  # 96 * 4h = 16 days, enough to get previous day
            # Get index of 4h bars that belong to previous day
            # Simplified: use 24 bars back (6 * 4h = 24h) for previous day's OHLC
            prev_day_idx = i - 24
            if prev_day_idx >= 24:  # ensure we have enough data for previous day
                # Get high, low, close of the previous day (24h period)
                day_high = np.max(high[prev_day_idx-24:prev_day_idx])
                day_low = np.min(low[prev_day_idx-24:prev_day_idx])
                day_close = close[prev_day_idx-1]  # close of previous day
                
                # Calculate Camarilla levels
                range_val = day_high - day_low
                if range_val > 0:
                    camarilla_r3 = day_close + (range_val * 1.1 / 4)
                    camarilla_s3 = day_close - (range_val * 1.1 / 4)
                else:
                    camarilla_r3 = curr_high
                    camarilla_s3 = curr_low
            else:
                camarilla_r3 = curr_high
                camarilla_s3 = curr_low
        else:
            camarilla_r3 = curr_high
            camarilla_s3 = curr_low
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Long: break above Camarilla R3 AND uptrend AND volume spike
            long_condition = curr_close > camarilla_r3 and curr_close > ema_50 and volume_spike
            # Short: break below Camarilla S3 AND downtrend AND volume spike
            short_condition = curr_close < camarilla_s3 and curr_close < ema_50 and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or price falls below EMA50
            if curr_close <= entry_price - 2.0 * atr_val or curr_close < ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or price rises above EMA50
            if curr_close >= entry_price + 2.0 * atr_val or curr_close > ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0