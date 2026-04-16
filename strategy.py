#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d ADX trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 level AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period average.
# Short when price breaks below Camarilla S3 level AND 1d ADX > 25 AND volume > 1.5x 20-period average.
# Exit on opposite Camarilla level (R3/S3) or ATR-based stoploss (2*ATR from entry).
# Uses discrete position size 0.25. Camarilla levels provide intraday support/resistance that work in ranging markets,
# while ADX filter ensures we only trade when there's sufficient trend strength to avoid false breakouts.
# Volume confirmation adds validity to breakouts. Designed for 4h timeframe to capture medium-term moves.
# Works in both bull and bear markets by requiring ADX > 25 (strong trend) and volume confirmation.
# Target: 75-200 total trades over 4 years (19-50/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Camarilla Pivot Levels (based on previous day) ===
    # Camarilla levels calculated from previous day's OHLC
    # We'll use the previous 4h bar's high/low/close to calculate for current bar
    # Actually, for intraday, we use the previous day's daily OHLC
    # But since we're on 4h, we need to get the daily OHLC from 1d data
    
    # === 1d Indicators: Get daily OHLC for Camarilla calculation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from daily OHLC
    # Camarilla formulas:
    # H4 = close + 1.1*(high - low)/1.2
    # H3 = close + 1.1*(high - low)/2.4
    # H2 = close + 1.1*(high - low)/4
    # H1 = close + 1.1*(high - low)/6
    # L1 = close - 1.1*(high - low)/6
    # L2 = close - 1.1*(high - low)/4
    # L3 = close - 1.1*(high - low)/2.4
    # L4 = close - 1.1*(high - low)/1.2
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 2.4
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 2.4
    camarilla_h4 = close_1d + 1.1 * (high_1d - low_1d) / 1.2
    camarilla_l4 = close_1d - 1.1 * (high_1d - low_1d) / 1.2
    
    # Align to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # === 1d Indicators: ADX for trend strength ===
    # ADX calculation: +DI, -DI, DX, then ADX smoothed
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).diff()
    tr2 = pd.Series(low_1d).diff().abs()
    tr3 = pd.Series(close_1d).shift(1).diff().abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    
    # +DM and -DM
    dm_plus = pd.Series(high_1d).diff()
    dm_minus = pd.Series(low_1d).diff().abs()
    dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
    dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # +DI and -DI
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where(np.isnan(dx), 0, dx)
    
    # ADX (smoothed DX)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    adx_strong = adx_1d_aligned > 25  # Strong trend filter
    
    # === 1d Indicators: Volume Spike ===
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
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
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or np.isnan(adx_1d_aligned[i]) or
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
            # Exit if price breaks below Camarilla L3
            if price < camarilla_l3_aligned[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Camarilla H3
            if price > camarilla_h3_aligned[i]:
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
            # LONG: Price breaks above Camarilla H3 AND ADX > 25 AND volume spike
            if price > camarilla_h3_aligned[i] and adx_strong[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Camarilla L3 AND ADX > 25 AND volume spike
            elif price < camarilla_l3_aligned[i] and adx_strong[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Camarilla_H3L3_1dADX25_VolumeSpike_V1"
timeframe = "4h"
leverage = 1.0