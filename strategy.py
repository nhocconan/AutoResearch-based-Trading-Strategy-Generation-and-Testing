#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot R1/S1 breakout with 4h EMA34 trend filter and volume spike
# Long when price breaks above Camarilla R1 + 4h EMA34 uptrend + volume > 1.8x 20-period avg
# Short when price breaks below Camarilla S1 + 4h EMA34 downtrend + volume > 1.8x 20-period avg
# Uses discrete position sizing (0.20) to control drawdown and minimize fee drag.
# 4h EMA34 provides strong trend filter reducing whipsaws in both bull and bear markets.
# Volume threshold (1.8x) targets ~15-35 trades/year on 1h timeframe to avoid overtrading.
# Session filter (08-20 UTC) reduces noise trades.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # === 4h Indicator: EMA34 ===
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # === 1h Camarilla Pivot Levels (based on prior bar) ===
    # Pivot = (H + L + C) / 3
    # R1 = Pivot + (H - L) * 1.1 / 12
    # S1 = Pivot - (H - L) * 1.1 / 12
    # Using prior bar's OHLC to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    camarilla_r1 = pivot + (prev_high - prev_low) * 1.1 / 12.0
    camarilla_s1 = pivot - (prev_high - prev_low) * 1.1 / 12.0
    
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
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.8x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.8)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R1 (close > R1)
        # 2. 4h EMA34 uptrend (close > EMA34)
        # 3. Volume confirmation
        if (close[i] > camarilla_r1[i]) and \
           (close[i] > ema_34_4h_aligned[i]) and vol_confirm:
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S1 (close < S1)
        # 2. 4h EMA34 downtrend (close < EMA34)
        # 3. Volume confirmation
        elif (close[i] < camarilla_s1[i]) and \
             (close[i] < ema_34_4h_aligned[i]) and vol_confirm:
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_Camarilla_R1S1_4hEMA34_Volume_Filter_v1"
timeframe = "1h"
leverage = 1.0