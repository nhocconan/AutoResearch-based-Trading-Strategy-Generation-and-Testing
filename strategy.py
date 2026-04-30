#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above R3 + price > 1d EMA34 + volume > 1.5x 20-bar average.
# Short when price breaks below S3 + price < 1d EMA34 + volume spike.
# Uses ATR trailing stop (2.0x) for risk management.
# Targets 50-150 total trades over 4 years (12-37/year) with discrete position sizing (0.25).
# Camarilla levels provide precise intraday support/resistance; EMA34 filters trend direction; volume confirms breakout strength.
# Works in both bull and bear markets by aligning with higher-timeframe trend.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We use daily high/low/close to calculate levels for current 12h session
    # For simplicity, we'll use rolling window of 28 bars (2 days of 12h data) to approximate daily OHLC
    lookback = 28  # 2 days * 2 periods per day (12h)
    roll_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    roll_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    roll_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).last().values
    
    # Camarilla R3 and S3
    rang = roll_high - roll_low
    r3 = roll_close + 1.1 * rang
    s3 = roll_close - 1.1 * rang
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # ATR for trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(50, lookback, 20)  # warmup for EMA34, Camarilla, and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if np.isnan(ema_34_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            continue
        
        # Regime filter: price above/below 1d EMA34 determines trend direction
        is_uptrend = close[i] > ema_34_aligned[i]
        is_downtrend = close[i] < ema_34_aligned[i]
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_volume_confirm = volume_confirm[i]
        curr_r3 = r3[i]
        curr_s3 = s3[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3 + uptrend + volume confirmation
            if curr_close > curr_r3 and is_uptrend and curr_volume_confirm:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_close
            # Short: price breaks below S3 + downtrend + volume spike
            elif curr_close < curr_s3 and is_downtrend and curr_volume_confirm:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_close
        
        elif position == 1:  # Long position
            # Update highest high since entry
            if curr_high > highest_since_entry:
                highest_since_entry = curr_high
            
            # Trailing stop: 2.0 * ATR below highest since entry
            if curr_close < highest_since_entry - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_low < lowest_since_entry:
                lowest_since_entry = curr_low
            
            # Trailing stop: 2.0 * ATR above lowest since entry
            if curr_close > lowest_since_entry + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals