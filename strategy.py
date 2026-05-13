#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R1/S1 breakout with 4h trend filter (HMA21) and 1d volume spike confirmation.
# Long when price breaks above Camarilla R1 level, 4h HMA21 is rising (trending up), and 1d volume > 2.0x 20-period average.
# Short when price breaks below Camarilla S1 level, 4h HMA21 is falling (trending down), and 1d volume > 2.0x 20-period average.
# Uses ATR(14) trailing stop (1.5x) for risk control.
# Session filter: 08-20 UTC to avoid low-volume periods.
# Target: 60-150 total trades over 4 years (15-37/year) on 1h.
# Uses discrete position size 0.20 to minimize fee churn.

name = "1h_Camarilla_R1S1_Breakout_4hHMA21_Trend_1dVolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

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
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla levels (based on previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R1, S1
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align Camarilla levels to 1h timeframe (wait for 1d bar to close)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Get 4h data for HMA21 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate HMA(21) on 4h data
    # HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights / weights.sum(), mode='valid')
    
    wma_half = wma(close_4h, half_len)
    wma_full = wma(close_4h, 21)
    hma_21 = 2 * wma_half - wma_full
    hma_21 = wma(hma_21, sqrt_len)
    # Pad to original length
    hma_21_padded = np.full_like(close_4h, np.nan)
    hma_21_padded[half_len:-sqrt_len+1 if sqrt_len>1 else None] = hma_21[half_len-1:len(hma_21)-sqrt_len+1] if sqrt_len>1 else hma_21
    # Simple fallback: use EMA if HMA calculation fails
    if np.all(np.isnan(hma_21_padded)):
        hma_21_padded = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align 4h HMA21 to 1h timeframe (wait for 4h bar to close)
    hma_21_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_21_padded)
    
    # Calculate 1d volume confirmation: volume > 2.0x 20-period average
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_1d > (2.0 * vol_ma_20_1d)
    
    # Align 1d volume spike to 1h timeframe (wait for 1d bar to close)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN or outside session
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(hma_21_4h_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(volume_spike_1d_aligned[i]) or not in_session[i]):
            signals[i] = 0.0
            # Carry forward tracking values when flat
            if i > 0 and position == 0:
                highest_since_entry[i] = highest_since_entry[i-1]
                lowest_since_entry[i] = lowest_since_entry[i-1]
            continue
        
        if position == 0:
            # LONG: Price > Camarilla R1 AND 4h HMA21 rising (trending up) AND 1d volume spike
            if close[i] > camarilla_r1_aligned[i] and hma_21_4h_aligned[i] > hma_21_4h_aligned[i-1] and volume_spike_1d_aligned[i]:
                signals[i] = 0.20
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Price < Camarilla S1 AND 4h HMA21 falling (trending down) AND 1d volume spike
            elif close[i] < camarilla_s1_aligned[i] and hma_21_4h_aligned[i] < hma_21_4h_aligned[i-1] and volume_spike_1d_aligned[i]:
                signals[i] = -0.20
                position = -1
                lowest_since_entry[i] = low[i]  # Initialize tracking
            else:
                signals[i] = 0.0
                # Carry forward tracking values when flat
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
        elif position == 1:
            # Update highest high since entry
            highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            # EXIT LONG: trailing stop hit (1.5x ATR)
            trailing_stop = close[i] < (highest_since_entry[i] - 1.5 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                highest_since_entry[i] = np.nan
            else:
                signals[i] = 0.20
                # Carry forward tracking
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            # EXIT SHORT: trailing stop hit (1.5x ATR)
            trailing_stop = close[i] > (lowest_since_entry[i] + 1.5 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = -0.20
                # Carry forward tracking
                if i > 0:
                    lowest_since_entry[i] = lowest_since_entry[i-1]
    
    return signals