#!/usr/bin/env python3
"""
1d_Weekly_Camarilla_R1_S1_Breakout_WeeklyTrend_Filter_v1
Hypothesis: Trade daily Camarilla R3/S3 breakouts with weekly EMA trend filter and volume confirmation. Designed for low trade frequency (~15-25/year) by requiring confluence: breakout at major daily pivot + weekly trend + volume spike (>1.5x average). Works in bull markets (long breakouts with uptrend) and bear markets (short breakdowns against weekly downtrend). Uses ATR trailing stop (2.0) for risk control. Focus on BTC/ETH as primary targets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA(34) for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get daily data for Camarilla pivot levels (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous daily bar
    # R3 = close + 1.1*(high-low)*1.1/4
    # S3 = close - 1.1*(high-low)*1.1/4
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 4
    
    # Align weekly EMA and daily Camarilla levels to daily timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: 1.5x average volume (20-day)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop (14-period on daily)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    # Warmup: max of weekly EMA (34), daily data (1), volume MA (20), daily ATR (14)
    start_idx = max(34, 1, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr_14[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_34_1w_val = ema_34_1w_aligned[i]
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        atr_14_val = atr_14[i]
        
        if position == 0:
            # Long: break above R3, weekly uptrend (close > EMA34), volume spike
            long_signal = (high_val > camarilla_r3_val) and (close_val > ema_34_1w_val) and (volume_val > 1.5 * vol_ma_val)
            # Short: break below S3, weekly downtrend (close < EMA34), volume spike
            short_signal = (low_val < camarilla_s3_val) and (close_val < ema_34_1w_val) and (volume_val > 1.5 * vol_ma_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                long_stop = entry_price - 2.0 * atr_14_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                short_stop = entry_price + 2.0 * atr_14_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Update trailing stop: move stop up as price makes new highs
            long_stop = max(long_stop, high_val - 2.0 * atr_14_val)
            # Exit: trailing stop hit or weekly trend reversal (close < EMA34)
            if (low_val < long_stop) or (close_val < ema_34_1w_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Update trailing stop: move stop down as price makes new lows
            short_stop = min(short_stop, low_val + 2.0 * atr_14_val)
            # Exit: trailing stop hit or weekly trend reversal (close > EMA34)
            if (high_val > short_stop) or (close_val > ema_34_1w_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Weekly_Camarilla_R1_S1_Breakout_WeeklyTrend_Filter_v1"
timeframe = "1d"
leverage = 1.0