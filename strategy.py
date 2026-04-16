#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla R1/S1 breakout with 1d ADX(14) trend filter and volume confirmation.
# Long when price breaks above 4h Camarilla R1 AND 1d ADX > 25 (trending) AND 1h volume > 1.5x 20-period average.
# Short when price breaks below 4h Camarilla S1 AND 1d ADX > 25 AND 1h volume > 1.5x 20-period average.
# Exit on opposite Camarilla break (S1 for longs, R1 for shorts) or ATR(14) stoploss (2*ATR from entry).
# Uses discrete position size 0.20. Designed to capture intraday momentum in trending markets with volume confirmation.
# Target: 60-150 total trades over 4 years (15-37/year) to balance edge and fee drag for 1h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Camarilla Pivots (R1, S1) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Typical price for 4h bar
    typical_4h = (high_4h + low_4h + close_4h) / 3.0
    # Camarilla width: (high - low) * 1.1 / 12
    width_4h = (high_4h - low_4h) * 1.1 / 12.0
    camarilla_r1_4h = close_4h + width_4h * 1.1
    camarilla_s1_4h = close_4h - width_4h * 1.1
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1_4h)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1_4h)
    
    # === 1d Indicators: ADX(14) for trend strength ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).diff()
    tr2 = pd.Series(low_1d).diff().abs()
    tr3 = pd.Series(close_1d).shift(1).diff().abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = pd.Series(high_1d).diff()
    dm_minus = pd.Series(low_1d).diff().abs()
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0.0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0.0)
    
    # Smoothed TR, DM+, DM-
    tr_14 = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    adx_trending = adx_aligned > 25  # Trending market filter
    
    # === 1h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # === 1h ATR for stoploss ===
    tr1_h = pd.Series(high).diff()
    tr2_h = pd.Series(low).diff().abs()
    tr3_h = pd.Series(close).shift(1).diff().abs()
    tr_h = pd.concat([tr1_h, tr2_h, tr3_h], axis=1).max(axis=1)
    atr_1h_raw = pd.Series(tr_h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for ADX/ATR)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr_1h_raw[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_1h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Camarilla S1
            if price < camarilla_s1_aligned[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Camarilla R1
            if price > camarilla_r1_aligned[i]:
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
            # LONG: Price breaks above Camarilla R1 AND ADX trending AND volume spike
            if price > camarilla_r1_aligned[i] and adx_trending[i] and vol_spike:
                signals[i] = 0.20
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Camarilla S1 AND ADX trending AND volume spike
            elif price < camarilla_s1_aligned[i] and adx_trending[i] and vol_spike:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_Camarilla_R1_S1_1dADX25_VolumeSpike_V1"
timeframe = "1h"
leverage = 1.0