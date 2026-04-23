#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R3/S3 breakout + 4h EMA20 trend filter + volume spike.
Long when price breaks above Camarilla R3 AND close > 4h EMA20 AND volume > 1.5x 20-period average.
Short when price breaks below Camarilla S3 AND close < 4h EMA20 AND volume > 1.5x 20-period average.
Exit when price crosses Camarilla pivot point (mean reversion) or ATR stoploss hit.
Uses discrete position sizing (0.20) to minimize fee churn. Targets 15-37 trades/year per symbol.
Camarilla levels provide precise intraday support/resistance, while 4h EMA20 ensures alignment with higher-timeframe trend.
Volume confirmation filters weak breakouts. Designed for 1h timeframe with session filter (08-20 UTC) to reduce noise.
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
    
    # Load 4h data for EMA20 trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA20 for trend filter
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align HTF EMA20 to 1h timeframe
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Calculate Camarilla levels on 1h timeframe using previous bar's OHLC
    # Camarilla: Pivot = (H+L+C)/3, R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 2.0
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 2.0
    camarilla_pivot = pivot  # Use pivot as mean reversion exit level
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(100, 20, 14)  # Ensure warmup for EMA20, ATR(14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: Camarilla breakout above R3 AND 4h EMA20 uptrend AND volume spike
            if (price > camarilla_r3[i] and 
                close[i] > ema20_4h_aligned[i] and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.20
                position = 1
                entry_price = price
            # Short: Camarilla breakout below S3 AND 4h EMA20 downtrend AND volume spike
            elif (price < camarilla_s3[i] and 
                  close[i] < ema20_4h_aligned[i] and 
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.20
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: price crosses Camarilla pivot (mean reversion)
            if position == 1 and price < camarilla_pivot[i]:
                exit_signal = True
            elif position == -1 and price > camarilla_pivot[i]:
                exit_signal = True
            
            # ATR-based stoploss: 2.0 * ATR from entry
            if position == 1 and price < entry_price - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > entry_price + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R3S3_4hEMA20_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0