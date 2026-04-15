#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot R4/S4 breakout with 1w EMA50 trend filter and volume spike
# Long when price breaks above Camarilla R4 + 1w EMA50 uptrend + volume > 2.0x 20-period avg
# Short when price breaks below Camarilla S4 + 1w EMA50 downtrend + volume > 2.0x 20-period avg
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# 1w EMA50 provides strong trend filter reducing whipsaws in both bull and bear markets.
# Volume threshold (2.0x) and Camarilla R4/S4 (extreme levels) target ~15-30 trades/year on 12h timeframe to avoid overtrading.
# Camarilla pivots calculated from prior 12h bar's high/low/close for structure-based entries.

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
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w Indicator: EMA50 ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 12h Camarilla Pivot Levels (based on prior bar) ===
    # Pivot = (H + L + C) / 3
    # R4 = Pivot + (H - L) * 1.1 / 2
    # S4 = Pivot - (H - L) * 1.1 / 2
    # Using prior bar's OHLC to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    camarilla_r4 = pivot + (prev_high - prev_low) * 1.1 / 2.0
    camarilla_s4 = pivot - (prev_high - prev_low) * 1.1 / 2.0
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 20) + 5  # EMA50 + Donchian(20) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r4[i]) or np.isnan(camarilla_s4[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R4 (close > R4)
        # 2. 1w EMA50 uptrend (close > EMA50)
        # 3. Volume confirmation
        if (close[i] > camarilla_r4[i]) and \
           (close[i] > ema_50_1w_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S4 (close < S4)
        # 2. 1w EMA50 downtrend (close < EMA50)
        # 3. Volume confirmation
        elif (close[i] < camarilla_s4[i]) and \
             (close[i] < ema_50_1w_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R4S4_1wEMA50_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0