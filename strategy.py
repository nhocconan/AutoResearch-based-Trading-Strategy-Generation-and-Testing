#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter (HMA21) and volume confirmation.
# Long when price breaks above Camarilla R1 AND 4h HMA21 trending up AND volume > 1.5x 20-period average.
# Short when price breaks below Camarilla S1 AND 4h HMA21 trending down AND volume > 1.5x 20-period average.
# Exit on opposite Camarilla level (S1 for longs, R1 for shorts) or ATR-based stop (1.5*ATR).
# Uses discrete position size 0.20. Session filter 08-20 UTC to reduce noise.
# Designed for 1h timeframe with HTF direction from 4h to avoid false breakouts in ranging markets.
# Target: 80-120 total trades over 4 years (20-30/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1h Indicators: Camarilla Pivots (based on previous bar) ===
    # Camarilla levels calculated from previous bar's OHLC
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # avoid NaN on first bar
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    # Calculate pivot and ranges
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    camarilla_r1 = pivot + (range_hl * 1.1 / 12.0)  # R1
    camarilla_s1 = pivot - (range_hl * 1.1 / 12.0)  # S1
    camarilla_r2 = pivot + (range_hl * 1.1 / 6.0)   # R2
    camarilla_s2 = pivot - (range_hl * 1.1 / 6.0)   # S2
    
    # === 4h Indicators: HMA(21) for trend ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    # HMA calculation: WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    half_len = 10  # 21/2 ≈ 10
    sqrt_len = 4   # sqrt(21) ≈ 4.58
    wma_half = pd.Series(close_4h).ewm(span=half_len, adjust=False, min_periods=half_len).mean().values
    wma_full = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    raw_hma = 2 * wma_half - wma_full
    hma_4h = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False, min_periods=sqrt_len).mean().values
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_up = hma_4h_aligned > np.roll(hma_4h_aligned, 1)
    hma_down = hma_4h_aligned < np.roll(hma_4h_aligned, 1)
    
    # === 4h Indicators: ATR(14) for stoploss ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_arr = df_4h['close'].values
    tr1 = pd.Series(high_4h).diff()
    tr2 = pd.Series(low_4h).diff().abs()
    tr3 = pd.Series(close_4h_arr).shift(1).diff().abs()
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h_raw = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h_raw)
    
    # === 1h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 40 periods needed)
    warmup = 50
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or np.isnan(hma_4h_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr_4h_aligned[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_4h_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Camarilla S1
            if price < camarilla_s1[i]:
                exit_signal = True
            # ATR-based stoploss: 1.5*ATR below entry
            elif price < entry_price - 1.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Camarilla R1
            if price > camarilla_r1[i]:
                exit_signal = True
            # ATR-based stoploss: 1.5*ATR above entry
            elif price > entry_price + 1.5 * atr_val:
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
                signals[i] = 0.20
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Camarilla S1 AND HMA trending down AND volume spike
            elif price < camarilla_s1[i] and hma_down[i] and vol_spike:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_Camarilla_R1_S1_4hHMA21_VolumeSpike_V1"
timeframe = "1h"
leverage = 1.0