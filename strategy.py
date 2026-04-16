#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA34 trend filter and volume spike
# Long when price breaks above Camarilla R1 + 4h EMA34 uptrend + volume > 2.0x 20-period avg
# Short when price breaks below Camarilla S1 + 4h EMA34 downtrend + volume > 2.0x 20-period avg
# Uses discrete position sizing (0.20) to control drawdown and minimize fee drag.
# 4h EMA34 provides strong trend filter reducing whipsaws in both bull and bear markets.
# Volume threshold (2.0x) targets ~15-30 trades/year on 1h timeframe to avoid overtrading.
# Camarilla pivot levels provide intraday structure-based breakout levels that work in ranging and trending markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # === 4h Indicator: EMA34 ===
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # === Daily Pivot Points (Camarilla) ===
    # Using previous day's OHLC to calculate today's levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align to 1h timeframe (previous day's levels available at open)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(34, 20) + 5  # EMA34 + Donchian(20) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R1 (close > R1)
        # 2. 4h EMA34 uptrend (close > EMA34)
        # 3. Volume confirmation
        if (close[i] > camarilla_r1_aligned[i]) and \
           (close[i] > ema_34_4h_aligned[i]) and vol_confirm:
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S1 (close < S1)
        # 2. 4h EMA34 downtrend (close < EMA34)
        # 3. Volume confirmation
        elif (close[i] < camarilla_s1_aligned[i]) and \
             (close[i] < ema_34_4h_aligned[i]) and vol_confirm:
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_Camarilla_R1_S1_4hEMA34_Volume_Filter_v1"
timeframe = "1h"
leverage = 1.0