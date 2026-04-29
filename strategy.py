#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation
# Uses Camarilla pivot levels (R3, S3) from 1h data for precise entry/exit
# Long when price breaks above R3 with volume spike AND 4h EMA50 uptrend
# Short when price breaks below S3 with volume spike AND 4h EMA50 downtrend
# Volume confirmation reduces false breaks. Works in both bull/bear by following 4h trend.
# Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_VolumeSpike_v1"
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
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load HTF data ONCE before loop for 4h calculations
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h Camarilla pivot points (using previous day's high/low/close)
    # For intraday, we use rolling window of 24 periods (24*1h = 1 day)
    lookback = 24
    if len(high) < lookback:
        return np.zeros(n)
    
    # Rolling max/min/close for pivot calculation
    roll_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    roll_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    roll_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).last().values
    
    pivot = (roll_high + roll_low + roll_close) / 3
    range_hl = roll_high - roll_low
    
    # Camarilla levels
    r3 = pivot + (range_hl * 1.1 / 4)
    s3 = pivot - (range_hl * 1.1 / 4)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = lookback  # warmup for pivot calculation
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if HTF data not available
        if np.isnan(ema50_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r3 = r3[i]
        curr_s3 = s3[i]
        curr_ema50 = ema50_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Trend regime: bullish if price > 4h EMA50, bearish if price < 4h EMA50
        is_bullish_regime = curr_close > curr_ema50
        is_bearish_regime = curr_close < curr_ema50
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and in session
            if curr_volume_confirm:
                # Bullish entry: price breaks above R3 AND bullish regime
                if curr_close > curr_r3 and is_bullish_regime:
                    signals[i] = 0.20
                    position = 1
                # Bearish entry: price breaks below S3 AND bearish regime
                elif curr_close < curr_s3 and is_bearish_regime:
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price breaks below R3 OR regime changes to bearish
            if (curr_close < curr_r3) or (not is_bullish_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price breaks above S3 OR regime changes to bullish
            if (curr_close > curr_s3) or (not is_bearish_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals