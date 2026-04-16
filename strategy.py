#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ADX regime filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND 1d ADX > 25 (trending) AND 1d volume > 1.5x 20-period average.
# Short when price breaks below Donchian(20) low AND 1d ADX > 25 AND 1d volume > 1.5x 20-period average.
# Exit on ATR-based stoploss (2*ATR from entry) or opposite Donchian breakout.
# Uses discrete position size 0.25. Works in both bull and bear markets by requiring
# volume confirmation and trend strength via 1d ADX. Target: 75-200 total trades over 4 years (19-50/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Donchian(20) and ATR(14) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high = highest_high
    donchian_low = lowest_low
    
    # True Range for ATR
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h_raw = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # === 1d Indicators: ADX(14) and Volume Spike ===
    df_1d = get_htf_data(prices, '1d')
    
    # ADX calculation
    plus_dm = pd.Series(df_1d['high']).diff()
    minus_dm = pd.Series(df_1d['low']).diff().abs()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr1_1d = pd.Series(df_1d['high']).diff()
    tr2_1d = pd.Series(df_1d['low']).diff().abs()
    tr3_1d = pd.Series(df_1d['close']).shift(1).diff().abs()
    tr_1d = pd.concat([tr1_1d, tr2_1d, tr3_1d], axis=1).max(axis=1)
    
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    plus_di_1d = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr_1d)
    minus_di_1d = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr_1d)
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Volume spike
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for ADX)
    warmup = 60
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(atr_4h_raw[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(volume_spike[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        adx_val = adx_1d_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr_4h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Donchian low
            if price < donchian_low[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Donchian high
            if price > donchian_high[i]:
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
            # LONG: price breaks above Donchian high AND 1d ADX > 25 AND volume spike
            if (price > donchian_high[i] and adx_val > 25 and vol_spike):
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: price breaks below Donchian low AND 1d ADX > 25 AND volume spike
            elif (price < donchian_low[i] and adx_val > 25 and vol_spike):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Donchian20_1dADX_VolumeSpike_V1"
timeframe = "4h"
leverage = 1.0