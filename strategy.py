#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume spike confirmation.
Long when price breaks above R3 AND close > 4h EMA34 AND volume > 2.0x 20-period average.
Short when price breaks below S3 AND close < 4h EMA34 AND volume > 2.0x 20-period average.
Exit when price reverts to Camarilla Pivot point (PP) or ATR-based stoploss hits.
Uses discrete position sizing (0.20) to minimize fee churn. Targets 15-30 trades/year per symbol.
Camarilla R3/S3 levels provide fewer but higher-quality breakouts with volume confirmation.
4h EMA34 provides medium-term trend filter that adapts to both bull and bear markets.
Session filter (08-20 UTC) reduces noise trades during low-liquidity periods.
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load 4h data for Camarilla calculation and EMA34 - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Camarilla levels for 4h timeframe (using previous bar's OHLC)
    # Camarilla: PP = (H+L+C)/3, R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close = np.roll(close_4h, 1)
    
    # First bar: use current values (will be refined as more data comes)
    prev_high[0] = high_4h[0]
    prev_low[0] = low_4h[0]
    prev_close[0] = close_4h[0]
    
    camarilla_pp = (prev_high + prev_low + prev_close) / 3.0
    camarilla_range = prev_high - prev_low
    camarilla_r3 = prev_close + camarilla_range * 1.1 / 4.0
    camarilla_s3 = prev_close - camarilla_range * 1.1 / 4.0
    
    # Align 4h Camarilla levels to 1h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pp)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Calculate EMA34 on 4h close data
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 4h EMA34 to 1h timeframe
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Volume average (20-period) on 1h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) on 1h data for stoploss
    tr1 = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr2 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema34_4h_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R3 AND close > 4h EMA34 AND volume spike
            if (price > camarilla_r3_aligned[i] and 
                close[i] > ema34_4h_aligned[i] and 
                volume[i] > 2.0 * vol_ma_val):
                signals[i] = 0.20
                position = 1
                entry_price = price
            # Short: price breaks below S3 AND close < 4h EMA34 AND volume spike
            elif (price < camarilla_s3_aligned[i] and 
                  close[i] < ema34_4h_aligned[i] and 
                  volume[i] > 2.0 * vol_ma_val):
                signals[i] = -0.20
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to PP or ATR stoploss
                if price <= camarilla_pp_aligned[i]:
                    exit_signal = True
                elif price < entry_price - 2.5 * atr[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to PP or ATR stoploss
                if price >= camarilla_pp_aligned[i]:
                    exit_signal = True
                elif price > entry_price + 2.5 * atr[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R3S3_Breakout_4hEMA34_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0