#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with 1d volume spike and ADX regime filter.
# Long when price breaks above R1 AND 1d volume > 1.8x 20-period average AND 1d ADX < 25 (range).
# Short when price breaks below S1 AND 1d volume > 1.8x 20-period average AND 1d ADX < 25 (range).
# Exit on ATR-based stoploss (2.5*ATR from entry) or opposite Camarilla breakout.
# Uses discrete position size 0.28. Target: 50-150 total trades over 4 years (12-37/year).
# Works in both bull and bear markets by requiring volume confirmation and range regime (ADX<25).
# Avoids overtrading via tight entry conditions: Camarilla breakout + volume spike + low volatility regime.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: Camarilla levels (R1, S1) from previous day ===
    # Camarilla levels calculated from previous 1d OHLC
    df_1d = get_htf_data(prices, '1d')
    # Previous day's high, low, close for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla R1 and S1
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === 1d Indicators: Volume Spike and ADX for regime filter ===
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.8 * vol_ma_1d_aligned)
    
    # 1d ADX calculation
    plus_dm = pd.Series(df_1d['high']).diff()
    minus_dm = pd.Series(df_1d['low']).diff().abs()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr1 = pd.Series(df_1d['high']).diff()
    tr2 = pd.Series(df_1d['low']).diff().abs()
    tr3 = pd.Series(df_1d['close']).shift(1).diff().abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    plus_di_1d = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr_1d)
    minus_di_1d = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr_1d)
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 12h ATR for stoploss ===
    tr1_12h = pd.Series(high).diff()
    tr2_12h = pd.Series(low).diff().abs()
    tr3_12h = pd.Series(close).shift(1).diff().abs()
    tr_12h = pd.concat([tr1_12h, tr2_12h, tr3_12h], axis=1).max(axis=1)
    atr_12h_raw = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
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
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(atr_12h_raw[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        adx_val = adx_1d_aligned[i]
        atr_val = atr_12h_raw[i]
        
        # Regime filter: only trade in low volatility (range) markets
        in_range = adx_val < 25
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below S1 (opposite Camarilla level)
            if price < camarilla_s1_aligned[i]:
                exit_signal = True
            # ATR-based stoploss: 2.5*ATR below entry
            elif price < entry_price - 2.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above R1 (opposite Camarilla level)
            if price > camarilla_r1_aligned[i]:
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
        if position == 0 and in_range:
            # LONG: Price breaks above R1 AND volume spike
            if price > camarilla_r1_aligned[i] and vol_spike:
                signals[i] = 0.28
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below S1 AND volume spike
            elif price < camarilla_s1_aligned[i] and vol_spike:
                signals[i] = -0.28
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.28
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_ADXRange_V1"
timeframe = "12h"
leverage = 1.0