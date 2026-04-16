#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator (Jaw/Teeth/Lips) with 1d HMA(34) trend filter and volume confirmation.
# Long when Alligator Lips > Teeth > Jaw (bullish alignment) AND 1d HMA(34) trending up AND volume > 1.5x 20-period average.
# Short when Alligator Lips < Teeth < Jaw (bearish alignment) AND 1d HMA(34) trending down AND volume > 1.5x 20-period average.
# Exit when Alligator alignment reverses or ATR-based stoploss (2.5*ATR from entry) is hit.
# Uses discrete position size 0.28. Designed to catch strong trends with Alligator's smoothing and avoid whipsaws.
# Works in both bull and bear markets by requiring 1d trend filter and volume confirmation, reducing false signals.
# Target: 80-180 total trades over 4 years (20-45/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: Williams Alligator (13,8,5) ===
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    jaw = pd.Series(close).ewm(alpha=1/13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(close).ewm(alpha=1/8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(close).ewm(alpha=1/5, adjust=False, min_periods=5).mean().values
    
    # Alligator alignment: bullish (lips > teeth > jaw) or bearish (lips < teeth < jaw)
    bullish_align = (lips > teeth) & (teeth > jaw)
    bearish_align = (lips < teeth) & (teeth < jaw)
    
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
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for Alligator/HMA/ATR)
    warmup = 80
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(hma_1d_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(atr_12h_raw[i]) or
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
            # Exit if Alligator alignment turns bearish
            if bearish_align[i]:
                exit_signal = True
            # ATR-based stoploss: 2.5*ATR below entry
            elif price < entry_price - 2.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Alligator alignment turns bullish
            if bullish_align[i]:
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
            # LONG: Bullish Alligator alignment AND HMA trending up AND volume spike
            if bullish_align[i] and hma_up[i] and vol_spike:
                signals[i] = 0.28
                position = 1
                entry_price = price
            
            # SHORT: Bearish Alligator alignment AND HMA trending down AND volume spike
            elif bearish_align[i] and hma_down[i] and vol_spike:
                signals[i] = -0.28
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.28
    
    return signals

name = "12h_WilliamsAlligator_1dHMA34_VolumeSpike_V1"
timeframe = "12h"
leverage = 1.0