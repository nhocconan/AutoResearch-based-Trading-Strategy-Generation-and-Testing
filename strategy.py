#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA(34) trend filter and volume confirmation.
# Long when price breaks above 1d Camarilla R1 level AND 1d EMA(34) trending up AND volume > 1.5x 20-period average.
# Short when price breaks below 1d Camarilla S1 level AND 1d EMA(34) trending down AND volume > 1.5x 20-period average.
# Exit on opposite Camarilla level touch (R1 for shorts, S1 for longs) or ATR-based stoploss (2*ATR from entry).
# Uses discrete position size 0.25. Designed to capture institutional level breaks with volume confirmation in trending markets.
# Works in both bull and bear markets by requiring 1d EMA trend filter and volume confirmation, avoiding false breakouts.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h_raw = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # === 1d Indicators: EMA(34) for trend ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    ema_1d_up = ema_1d_aligned > np.roll(ema_1d_aligned, 1)
    ema_1d_down = ema_1d_aligned < np.roll(ema_1d_aligned, 1)
    
    # === 1d Indicators: Camarilla Pivot Levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # Camarilla formula: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12
    camarilla_r1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
    camarilla_s1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 70 periods needed for EMA/Camarilla/volume)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(camarilla_r1_1d_aligned[i]) or 
            np.isnan(camarilla_s1_1d_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(atr_12h_raw[i]) or not session_filter[i]):
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
            # Exit if price touches or breaks below Camarilla S1 level
            if price <= camarilla_s1_1d_aligned[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price touches or breaks above Camarilla R1 level
            if price >= camarilla_r1_1d_aligned[i]:
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
            # LONG: Price breaks above Camarilla R1 AND EMA trending up AND volume spike
            if price > camarilla_r1_1d_aligned[i] and ema_1d_up[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Camarilla S1 AND EMA trending down AND volume spike
            elif price < camarilla_s1_1d_aligned[i] and ema_1d_down[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_Camarilla_R1S1_1dEMA34_VolumeSpike_V1"
timeframe = "12h"
leverage = 1.0