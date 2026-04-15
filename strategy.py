#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot R4/S4 breakout with 1d EMA50 trend filter and volume spike
# Long when price breaks above Camarilla R4 + 1d EMA50 uptrend + volume > 2.0x 24-period avg
# Short when price breaks below Camarilla S4 + 1d EMA50 downtrend + volume > 2.0x 24-period avg
# Uses discrete position sizing (0.30) to balance reward/risk and minimize fee drag.
# 1d EMA50 provides stronger trend filter than EMA34, reducing whipsaws in choppy markets.
# Volume threshold (2.0x) targets ~15-30 trades/year on 12h timeframe to avoid overtrading.
# Camarilla R4/S4 are stronger levels than R1/S1, requiring more significant breaks.

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
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicator: EMA50 ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
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
    
    # Volume SMA for confirmation (using 24-period for 12h timeframe)
    vol_sma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 24) + 5  # EMA50 + Donchian(24) + volume(24) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r4[i]) or np.isnan(camarilla_s4[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_sma_24[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 24-period volume SMA
        vol_confirm = volume[i] > (vol_sma_24[i] * 2.0)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R4 (close > R4)
        # 2. 1d EMA50 uptrend (close > EMA50)
        # 3. Volume confirmation
        if (close[i] > camarilla_r4[i]) and \
           (close[i] > ema_50_1d_aligned[i]) and vol_confirm:
            signals[i] = 0.30
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S4 (close < S4)
        # 2. 1d EMA50 downtrend (close < EMA50)
        # 3. Volume confirmation
        elif (close[i] < camarilla_s4[i]) and \
             (close[i] < ema_50_1d_aligned[i]) and vol_confirm:
            signals[i] = -0.30
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R4S4_1dEMA50_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0