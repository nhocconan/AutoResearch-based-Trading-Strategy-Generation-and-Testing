#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX(12) zero-cross with 1d HMA(34) trend filter, volume confirmation (1.5x 20-period average), and ATR(14) stoploss (2*ATR).
# Long when TRIX crosses above zero AND 1d HMA(34) trending up AND volume spike.
# Short when TRIX crosses below zero AND 1d HMA(34) trending down AND volume spike.
# Exit on ATR-based stoploss or opposite TRIX zero-cross.
# Uses discrete position size 0.25. TRIX is a momentum oscillator that filters noise and identifies trend changes.
# Volume confirmation avoids false breakouts. HMA(34) on 1d ensures alignment with higher timeframe trend.
# Designed for low trade frequency (<50/year) to minimize fee drag while capturing strong momentum moves.
# Works in both bull and bear markets by requiring 1d trend filter and volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: TRIX(12) ===
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) then percent change
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix_raw = 100 * (pd.Series(ema3).pct_change().values)
    # Smoothed TRIX signal line (9-period EMA of TRIX)
    trix_signal = pd.Series(trix_raw).ewm(span=9, adjust=False, min_periods=9).mean().values
    trix_cross_up = (trix_signal > 0) & (np.roll(trix_signal, 1) <= 0)
    trix_cross_down = (trix_signal < 0) & (np.roll(trix_signal, 1) >= 0)
    
    # === 1d Indicators: HMA(34) for trend ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # HMA calculation: WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    half_len = 17  # 34/2
    sqrt_len = 6   # sqrt(34) ≈ 5.83
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
    
    # === 4h ATR for stoploss ===
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr_4h = np.maximum(np.maximum(tr1, tr2), tr3)
    atr_4h_raw = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for TRIX/HMA/ATR)
    warmup = 60
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(trix_signal[i]) or np.isnan(hma_1d_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(atr_4h_raw[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_4h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if TRIX crosses below zero
            if trix_cross_down[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if TRIX crosses above zero
            if trix_cross_up[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR above entry
            elif price > entry_price + 2.0 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: TRIX crosses above zero AND HMA trending up AND volume spike
            if trix_cross_up[i] and hma_up[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: TRIX crosses below zero AND HMA trending down AND volume spike
            elif trix_cross_down[i] and hma_down[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_TRIX12_1dHMA34_VolumeSpike_V1"
timeframe = "4h"
leverage = 1.0