#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d ADX regime filter and volume confirmation.
# Bull Power = High - EMA(13), Bear Power = EMA(13) - Low.
# Long when Bull Power > 0 AND Bear Power rising AND ADX > 25 AND volume > 1.5x 20-period average.
# Short when Bear Power < 0 AND Bull Power falling AND ADX > 25 AND volume > 1.5x 20-period average.
# Exit on ATR-based stoploss (2*ATR from entry) or power divergence.
# Uses discrete position size 0.25. Works in both bull and bear markets by requiring
# ADX regime filter (trending markets only) and volume confirmation.
# Target: 75-200 total trades over 4 years (19-50/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: EMA13 for Elder Ray ===
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA(13)
    bull_power = high - ema_13
    # Bear Power = EMA(13) - Low
    bear_power = ema_13 - low
    
    # === 1d Indicators: ADX and Volume Spike ===
    df_1d = get_htf_data(prices, '1d')
    
    # ADX calculation (14-period)
    plus_dm = pd.Series(df_1d['high']).diff()
    minus_dm = pd.Series(df_1d['low']).diff().abs()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = pd.Series(df_1d['high']).diff()
    tr2 = pd.Series(df_1d['low']).diff().abs()
    tr3 = pd.Series(df_1d['close']).shift(1).diff().abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    plus_di_14 = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_14)
    minus_di_14 = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_14)
    dx_14 = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx_14 = pd.Series(dx_14).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Volume spike
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_1d)
    
    # === 6h ATR for stoploss ===
    tr1_6h = pd.Series(high).diff()
    tr2_6h = pd.Series(low).diff().abs()
    tr3_6h = pd.Series(close).shift(1).diff().abs()
    tr_6h = pd.concat([tr1_6h, tr2_6h, tr3_6h], axis=1).max(axis=1)
    atr_6h = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Align HTF indicators
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods needed for volume MA)
    warmup = 40
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(adx_14_aligned[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(atr_6h[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike_aligned[i]
        adx_val = adx_14_aligned[i]
        atr_val = atr_6h[i]
        
        # Power momentum (previous bar)
        bull_power_prev = bull_power[i-1] if i > 0 else 0
        bear_power_prev = bear_power[i-1] if i > 0 else 0
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Bull Power turns negative (momentum loss)
            if bull_power[i] <= 0 and bull_power_prev > 0:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Bear Power turns negative (momentum loss)
            if bear_power[i] <= 0 and bear_power_prev > 0:
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
            # LONG: Bull Power > 0 AND Bull Power rising AND ADX > 25 AND volume spike
            if (bull_power[i] > 0 and bull_power[i] > bull_power_prev and 
                adx_val > 25 and vol_spike):
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Bear Power > 0 AND Bear Power rising AND ADX > 25 AND volume spike
            elif (bear_power[i] > 0 and bear_power[i] > bear_power_prev and 
                  adx_val > 25 and vol_spike):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_ElderRay_1dADX_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0