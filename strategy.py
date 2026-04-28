#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Uses actual Camarilla pivot levels from daily timeframe for institutional reference points.
# 1d EMA34 ensures higher-timeframe trend alignment. Volume spike (>2.0x 20-bar average) confirms momentum.
# Designed for 12h timeframe to target 12-37 trades/year (50-150 total over 4 years) minimizing fee drag.
# Works in bull/bear markets by trading breakouts from key levels with volume confirmation and trend filter.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (based on previous day's OHLC)
    # Camarilla levels: R4, R3, R2, R1, PP, S1, S2, S3, S4
    # We focus on R3 and S3 for breakout entries
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate pivot point (PP) = (High + Low + Close) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Calculate range = High - Low
    range_1d = high_1d - low_1d
    # Camarilla R3 = PP + (Range * 1.1 / 4)
    r3 = pp + (range_1d * 1.1 / 4.0)
    # Camarilla S3 = PP - (Range * 1.1 / 4)
    s3 = pp - (range_1d * 1.1 / 4.0)
    
    # Align Camarilla levels to 12h timeframe (wait for daily close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(34, 20)  # EMA34, volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        price = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_34_val = ema_34_1d_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above R3 with volume confirmation and above 1d EMA34
            if price > r3_val and vol_confirm and price > ema_34_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: Price breaks below S3 with volume confirmation and below 1d EMA34
            elif price < s3_val and vol_confirm and price < ema_34_val:
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on stoploss or re-entry below R3
            # ATR-based stoploss: 2.5 * ATR below entry (using 12h ATR)
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price - 2.5 * atr_val
            # Exit on stoploss or price re-enters below R3 (breakout failed)
            if price < stop_loss or price < r3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on stoploss or re-entry above S3
            # ATR-based stoploss: 2.5 * ATR above entry
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price + 2.5 * atr_val
            # Exit on stoploss or price re-enters above S3 (breakout failed)
            if price > stop_loss or price > s3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals