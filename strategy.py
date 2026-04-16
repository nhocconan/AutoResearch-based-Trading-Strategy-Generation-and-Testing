#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d HMA(34) trend filter, volume confirmation (>1.4x 20-period average), and ATR(14) stoploss (2.5x ATR).
# Long when price breaks above Donchian upper band AND 1d HMA(34) trending up AND volume > 1.4x 20-period average.
# Short when price breaks below Donchian lower band AND 1d HMA(34) trending down AND volume > 1.4x 20-period average.
# Exit on ATR-based stoploss (2.5*ATR from entry) or opposite Donchian break.
# Uses discrete position size 0.28. Designed to capture strong momentum moves with volume confirmation in trending markets.
# Works in both bull and bear markets by requiring 1d trend filter (HMA direction) and volume confirmation, avoiding false breakouts.
# Target: 75-200 total trades over 4 years (19-50/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Donchian Channel (20) ===
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1d Indicators: HMA(34) for trend ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # HMA calculation: WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    half_len = 17  # 34/2
    sqrt_len = 5   # sqrt(34) ≈ 5.83
    wma_half = pd.Series(close_1d).ewm(span=half_len, adjust=False, min_periods=half_len).mean().values
    wma_full = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    raw_hma = 2 * wma_half - wma_full
    hma_1d = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False, min_periods=sqrt_len).mean().values
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_up = hma_1d_aligned > np.roll(hma_1d_aligned, 1)
    hma_down = hma_1d_aligned < np.roll(hma_1d_aligned, 1)
    
    # === 1d Indicators: Volume Spike (volume > 1.4x 20-period average) ===
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.4 * vol_ma_1d_aligned)
    
    # === 4h ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h_raw = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 60 periods needed for HMA/ATR/Donchian)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(hma_1d_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr_4h_raw[i]) or
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
            # Exit if price breaks below Donchian lower band
            if price < donchian_lower[i]:
                exit_signal = True
            # ATR-based stoploss: 2.5*ATR below entry
            elif price < entry_price - 2.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Donchian upper band
            if price > donchian_upper[i]:
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
            # LONG: Price breaks above Donchian upper AND HMA trending up AND volume spike
            if price > donchian_upper[i] and hma_up[i] and vol_spike:
                signals[i] = 0.28
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Donchian lower AND HMA trending down AND volume spike
            elif price < donchian_lower[i] and hma_down[i] and vol_spike:
                signals[i] = -0.28
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.28
    
    return signals

name = "4h_Donchian20_1dHMA34_VolumeSpike_V1"
timeframe = "4h"
leverage = 1.0