#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot R1/S1 breakout with 1d volume-weighted average price (VWAP) trend filter
# Long when price breaks above Camarilla R1 + close > 1d VWAP + volume spike
# Short when price breaks below Camarilla S1 + close < 1d VWAP + volume spike
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# 1d VWAP provides strong institutional trend filter that adapts to both bull and bear markets.
# Volume threshold (2.0x) targets ~15-35 trades/year on 12h timeframe to avoid overtrading.
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
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # === 1d Indicator: VWAP (Volume Weighted Average Price) ===
    # VWAP = cumulative(volume * price) / cumulative(volume)
    # Reset daily, so we need to compute it per day
    # For simplicity, we'll use a rolling approximation that resets at day boundaries
    # Using typical price * volume / volume cumulative
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d_values = vwap_1d.values
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d_values)
    
    # === 12h Camarilla Pivot Levels (based on prior bar) ===
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
    warmup = max(20, 1) + 5  # volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or
            np.isnan(vwap_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R1 (close > R1)
        # 2. Close > 1d VWAP (institutional bullish trend)
        # 3. Volume confirmation
        if (close[i] > camarilla_r1[i]) and \
           (close[i] > vwap_1d_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S1 (close < S1)
        # 2. Close < 1d VWAP (institutional bearish trend)
        # 3. Volume confirmation
        elif (close[i] < camarilla_s1[i]) and \
             (close[i] < vwap_1d_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R1S1_1dVWAP_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0