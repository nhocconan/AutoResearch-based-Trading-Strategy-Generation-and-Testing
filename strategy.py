#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeConfirm_v1
Hypothesis: Use daily timeframe with Camarilla R3/S3 breakout confirmed by weekly trend (price above/below weekly EMA34) and volume spike. Camarilla levels provide precise support/resistance that works in ranging and trending markets. Weekly EMA34 filters for higher-timeframe trend alignment. Volume spike confirms breakout validity. Targets 20-50 trades over 4 years (5-12/year) to minimize fee drag. Uses ATR-based stoploss for risk management.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate weekly EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly EMA34
    weekly_close = df_1w['close'].values
    ema34_1w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_trend_up = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate daily Camarilla pivot levels (R3, S3)
    # Based on previous day's OHLC
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    R3 = pivot + range_val * 1.1 / 2.0  # R3 = pivot + (high-low)*1.1/2
    S3 = pivot - range_val * 1.1 / 2.0  # S3 = pivot - (high-low)*1.1/2
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for all indicators
    start_idx = max(20, 34, 14)  # volume avg, weekly EMA, ATR
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or
            np.isnan(weekly_trend_up[i]) or np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        atr_val = atr[i]
        size = 0.25  # 25% position size
        
        if position == 0:
            # Flat - look for breakout with weekly trend and volume confirmation
            # Long: break above R3 + price above weekly EMA34 + volume spike
            long_entry = (close_val > R3[i]) and \
                       (close_val > weekly_trend_up[i]) and \
                       volume_spike[i]
            # Short: break below S3 + price below weekly EMA34 + volume spike
            short_entry = (close_val < S3[i]) and \
                        (close_val < weekly_trend_up[i]) and \
                        volume_spike[i]
            
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
            # Long - exit on S3 break or ATR stoploss
            exit_condition = (close_val < S3[i]) or \
                           (close_val < entry_price - 2.5 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on R3 break or ATR stoploss
            exit_condition = (close_val > R3[i]) or \
                           (close_val > entry_price + 2.5 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0