#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot (R1/S1) breakout with 1d HMA(34) trend filter and volume confirmation.
# Long when price breaks above Camarilla R1 AND 1d HMA(34) trending up AND volume > 1.5x 20-period average.
# Short when price breaks below Camarilla S1 AND 1d HMA(34) trending down AND volume > 1.5x 20-period average.
# Exit on opposite Camarilla level (S1 for longs, R1 for shorts) or ATR-based stoploss (2.5*ATR).
# Uses discrete position size 0.25. Designed to capture institutional breakouts with volume confirmation.
# Works in both bull and bear markets by requiring 1d trend filter and volume confirmation, avoiding false breakouts.
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: Camarilla Pivot Levels (based on previous day) ===
    # Calculate pivot from previous 12h bar's high, low, close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # avoid NaN on first bar
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    camarilla_r1 = pivot + (range_val * 1.1 / 12.0)  # R1 = pivot + (range * 1.1/12)
    camarilla_s1 = pivot - (range_val * 1.1 / 12.0)  # S1 = pivot - (range * 1.1/12)
    
    # === 1d Indicators: HMA(34) for trend ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # HMA calculation: WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    half_len = 17  # 34/2
    sqrt_len = int(np.sqrt(34))  # sqrt(34) ≈ 5.83 -> 5
    wma_half = pd.Series(close_1d).ewm(span=half_len, adjust=False, min_periods=half_len).mean().values
    wma_full = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    raw_hma = 2 * wma_half - wma_full
    hma_1d = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False, min_periods=sqrt_len).mean().values
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_up = hma_1d_aligned > np.roll(hma_1d_aligned, 1)
    hma_down = hma_1d_aligned < np.roll(hma_1d_aligned, 1)
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # === 12h ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h_raw = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 70 periods needed for HMA/ATR)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or np.isnan(hma_1d_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr_12h_raw[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_12h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Camarilla S1
            if price < camarilla_s1[i]:
                exit_signal = True
            # ATR-based stoploss: 2.5*ATR below entry
            elif price < entry_price - 2.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Camarilla R1
            if price > camarilla_r1[i]:
                exit_signal = True
            # ATR-based stoploss: 2.5*ATR above entry
            elif price > entry_price + 2.5 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Camarilla R1 AND HMA trending up AND volume spike
            if price > camarilla_r1[i] and hma_up[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Camarilla S1 AND HMA trending down AND volume spike
            elif price < camarilla_s1[i] and hma_down[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_Camarilla_R1_S1_1dHMA34_VolumeSpike_V1"
timeframe = "12h"
leverage = 1.0