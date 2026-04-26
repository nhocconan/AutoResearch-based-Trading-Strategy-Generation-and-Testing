#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation. 
In bull markets (price > 1d EMA34), go long on breakout above R3; in bear markets (price < 1d EMA34), go short on breakdown below S3.
Volume spike (current volume > 1.5 * 20-period average) confirms breakout strength.
Uses discrete sizing 0.25 to limit trades (~12-25/year) and ATR-based stoploss for risk control.
Designed to work in both bull and bear markets via 1d trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR for stoploss and volume average
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Volume average (20-period) for spike confirmation
    volume_series = pd.Series(volume)
    volume_avg = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Previous day's OHLC for Camarilla levels (from aligned 1d data)
    # We need the completed 1d bar's OHLC, so we use shift(1) on the 1d data before alignment
    prev_open_1d = df_1d['open'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    prev_close_1d = df_1d['close'].shift(1).values
    
    # Align previous day's OHLC to LTF (12h)
    prev_open_aligned = align_htf_to_ltf(prices, df_1d, prev_open_1d)
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high_1d)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low_1d)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close_1d)
    
    # Calculate Camarilla levels
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    camarilla_range = prev_high_aligned - prev_low_aligned
    r3 = prev_close_aligned + camarilla_range * 1.1 / 4
    s3 = prev_close_aligned - camarilla_range * 1.1 / 4
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 34 for 1d EMA, 20 for volume avg, and 1 for Camarilla (uses prev day)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_avg[i]) or
            np.isnan(r3[i]) or
            np.isnan(s3[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        volume_val = volume[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_avg_val = volume_avg[i]
        r3_val = r3[i]
        s3_val = s3[i]
        atr_val = atr[i]
        size = fixed_size
        
        # Volume spike confirmation: current volume > 1.5 * 20-period average
        volume_spike = volume_val > 1.5 * vol_avg_val
        
        if position == 0:
            # Flat - look for entry
            # Determine market regime from 1d EMA34
            is_uptrend = close_val > ema_34_val
            is_downtrend = close_val < ema_34_val
            
            # Long entry: price breaks above R3 in uptrend with volume spike
            long_entry = is_uptrend and close_val > r3_val and volume_spike
            # Short entry: price breaks below S3 in downtrend with volume spike
            short_entry = is_downtrend and close_val < s3_val and volume_spike
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on trend reversal or ATR-based stop
            # Exit if trend turns bearish OR price drops 2.5*ATR from entry
            if close_val < ema_34_val or close_val < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal or ATR-based stop
            # Exit if trend turns bullish OR price rises 2.5*ATR from entry
            if close_val > ema_34_val or close_val > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0