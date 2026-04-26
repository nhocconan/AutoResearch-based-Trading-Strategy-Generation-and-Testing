#!/usr/bin/env python3
"""
1h_Camarilla_R3S3_Breakout_4hTrend_1dRegimeFilter_Volume
Hypothesis: Use 4h EMA50 for trend direction, 1d Choppiness Index (CHOP < 40) for trending regime filter, and 1h Camarilla R3/S3 breakout with volume confirmation (>1.5x average volume). Enter on breakouts aligned with 4h trend and 1d trending regime. Uses discrete position sizing (0.20) to limit fee churn. Designed for 1h timeframe to capture intermediate swings while avoiding overtrading via strict HTF filters.
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
    
    # Load 4h data for HTF trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data for regime filter (Choppiness Index)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate True Range for 1d CHOP
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]
    
    # Choppiness Index (14-period)
    atr_sum = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denominator = max_high - min_low
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)
    chop_ratio = atr_sum / chop_denominator
    chop = 100 * np.log10(chop_ratio) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 1h average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.20
    
    # Start after warmup (need 20 for Camarilla, 50 for 4h EMA, 20 for volume, 14 for CHOP)
    start_idx = max(20, 50, 20, 14)
    
    for i in range(start_idx, n):
        # Need previous hour's OHLC for Camarilla levels
        if i < 1:
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
            
        # Previous hour's high, low, close (for Camarilla calculation)
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        # Calculate Camarilla levels
        range_val = prev_high - prev_low
        if range_val <= 0:
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
            
        # Camarilla R3 and S3 levels
        r3 = prev_close + range_val * 1.1 / 4
        s3 = prev_close - range_val * 1.1 / 4
        
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_4h_val = ema_50_4h_aligned[i]
        chop_val = chop_aligned[i]
        
        # Skip if any data not ready
        if np.isnan(r3) or np.isnan(s3) or np.isnan(ema_4h_val) or np.isnan(avg_vol) or np.isnan(chop_val):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = vol > 1.5 * avg_vol
        
        # Regime filter: 1d CHOP < 40 = trending market
        regime_ok = chop_val < 40
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        session_ok = 8 <= hour <= 20
        
        # Long logic: price breaks above R3 with 4h uptrend, volume confirmation, trending regime, and session
        long_condition = (close_val > r3) and (close_val > ema_4h_val) and volume_confirmed and regime_ok and session_ok
        # Short logic: price breaks below S3 with 4h downtrend, volume confirmation, trending regime, and session
        short_condition = (close_val < s3) and (close_val < ema_4h_val) and volume_confirmed and regime_ok and session_ok
        
        # Exit logic: 4h trend reversal OR regime change to ranging OR session end
        exit_long = (close_val < ema_4h_val) or (chop_val >= 40) or (hour < 8 or hour > 20)
        exit_short = (close_val > ema_4h_val) or (chop_val >= 40) or (hour < 8 or hour > 20)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1h_Camarilla_R3S3_Breakout_4hTrend_1dRegimeFilter_Volume"
timeframe = "1h"
leverage = 1.0