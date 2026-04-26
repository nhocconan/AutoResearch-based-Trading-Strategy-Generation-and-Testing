#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_HTFConfirm_v3
Hypothesis: Camarilla R3/S3 breakout on 4h with 1d EMA34 trend filter and volume confirmation. 
Uses tight entry conditions (price must close beyond R3/S3) to limit trades to 75-200 total over 4 years. 
Works in bull/bear markets via 1d trend filter (EMA34). Volume spike confirms institutional participation. 
ATR-based stoploss manages risk. Designed for BTC/ETH with SOL as secondary.
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
    
    # Load 1d data ONCE before loop for HTF trend filter and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d ATR(14) for stoploss
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate Camarilla levels from previous 1d OHLC
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), S3 = C - ((H-L)*1.1/4)
    # We need previous day's values, so shift by 1
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_range = prev_high - prev_low
    camarilla_R3 = prev_close + (camarilla_range * 1.1 / 4)
    camarilla_S3 = prev_close - (camarilla_range * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    
    # Volume spike: volume > 1.5x 20-period median volume
    volume_series = pd.Series(volume)
    vol_median_20 = volume_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (1.5 * vol_median_20)
    
    # Position size: 0.25 (25% of capital)
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 34 for 1d EMA, 14 for ATR, 20 for volume median, 1 for Camarilla shift
    start_idx = max(34, 14, 20, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(camarilla_R3_aligned[i]) or
            np.isnan(camarilla_S3_aligned[i]) or
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_34_val = ema_34_1d_aligned[i]
        atr_val = atr_14_1d_aligned[i]
        vol_spike = volume_spike[i]
        size = fixed_size
        camarilla_R3_val = camarilla_R3_aligned[i]
        camarilla_S3_val = camarilla_S3_aligned[i]
        
        if position == 0:
            # Flat - look for entry
            # Long entry: close > R3 and volume spike, in uptrend (close > EMA34_1d)
            long_entry = (close_val > camarilla_R3_val) and vol_spike and (close_val > ema_34_val)
            # Short entry: close < S3 and volume spike, in downtrend (close < EMA34_1d)
            short_entry = (close_val < camarilla_S3_val) and vol_spike and (close_val < ema_34_val)
            
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
            # Long - exit on trend reversal, ATR stoploss, or at S3 (mean reversion)
            stop_price = entry_price - 2.0 * atr_val
            if close_val < ema_34_val or close_val < stop_price or close_val < camarilla_S3_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal, ATR stoploss, or at R3 (mean reversion)
            stop_price = entry_price + 2.0 * atr_val
            if close_val > ema_34_val or close_val > stop_price or close_val > camarilla_R3_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_HTFConfirm_v3"
timeframe = "4h"
leverage = 1.0