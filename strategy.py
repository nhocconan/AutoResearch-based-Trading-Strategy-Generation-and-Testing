#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_ChopFilter_v1
Hypothesis: Camarilla R3/S3 breakout on 4h with 1d EMA34 trend filter, volume confirmation, and chop regime filter.
Breakouts above/below Camarilla R3/S3 levels capture strong momentum moves. Trend filter ensures we only trade
in direction of 1d trend to avoid counter-trend whipsaws. Volume spike confirms breakout authenticity.
Chop filter avoids sideways markets. Designed for 4h timeframe with target 75-200 trades over 4 years.
Uses discrete position sizing (0.30) to balance return and drawdown. Works in both bull and bear markets
by aligning with intermediate-term 1d trend and avoiding choppy regimes.
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
    
    # Calculate ATR for Camarilla levels (20-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels from previous 4h bar's OHLC
    prev_close = np.concatenate([[close[0]], close[:-1]])
    prev_high = np.concatenate([[high[0]], high[:-1]])
    prev_low = np.concatenate([[low[0]], low[:-1]])
    
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # Chop filter: choppiness index > 61.8 = choppy (avoid trading)
    # Calculate true range for chop
    tr_chop = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_chop[0] = high[0] - low[0]  # first bar
    atr_chop = pd.Series(tr_chop).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_chop * np.sqrt(14) / (max_high - min_low)) / np.log10(14)
    chop_filter = chop <= 61.8  # only trade when NOT choppy (<=61.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for ATR, EMA34, volume average, and chop
    start_idx = max(100, 34, 20, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(atr[i]) or np.isnan(chop_filter[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_trend = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        not_choppy = chop_filter[i]
        size = 0.30  # 30% position size
        
        if position == 0:
            # Flat - look for entry: breakout in direction of 1d trend with volume spike and not choppy
            # Long: price breaks above Camarilla R3 AND 1d trend is up (close > EMA34) AND volume spike AND not choppy
            # Short: price breaks below Camarilla S3 AND 1d trend is down (close < EMA34) AND volume spike AND not choppy
            long_breakout = close_val > camarilla_r3[i]
            short_breakout = close_val < camarilla_s3[i]
            trend_up = close_val > ema_trend
            trend_down = close_val < ema_trend
            
            if long_breakout and trend_up and vol_spike and not_choppy:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_breakout and trend_down and vol_spike and not_choppy:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit when price breaks below Camarilla S3 (failed breakout) or ATR stoploss hit
            if close_val < camarilla_s3[i] or close_val < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price breaks above Camarilla R3 (failed breakout) or ATR stoploss hit
            if close_val > camarilla_r3[i] or close_val > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0