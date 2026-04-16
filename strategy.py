#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d Supertrend(10,3) trend filter, volume confirmation, and ATR(14) stoploss.
# Long when price breaks above Donchian upper band AND 1d Supertrend is bullish AND volume > 1.5x 20-period average.
# Short when price breaks below Donchian lower band AND 1d Supertrend is bearish AND volume > 1.5x 20-period average.
# Exit on ATR-based stoploss (2*ATR from entry) or opposite Donchian break.
# Uses discrete position size 0.25. Supertrend provides adaptive trend filtering that works in both bull and bear markets.
# Volume confirmation avoids false breakouts. Target: 75-200 total trades over 4 years (19-50/year).

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
    
    # === 1d Indicators: Supertrend(10,3) for trend ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR for Supertrend
    tr1 = pd.Series(high_1d).diff()
    tr2 = pd.Series(low_1d).diff().abs()
    tr3 = pd.Series(close_1d).shift(1).diff().abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (high_1d + low_1d) / 2
    upper_band = hl2 + (3 * atr_1d)
    lower_band = hl2 - (3 * atr_1d)
    
    # Initialize bands
    final_upper = np.full_like(upper_band, np.nan)
    final_lower = np.full_like(lower_band, np.nan)
    supertrend = np.full_like(close_1d, np.nan)
    direction = np.full_like(close_1d, np.nan)  # 1 for uptrend, -1 for downtrend
    
    for i in range(10, len(close_1d)):
        if np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(atr_1d[i]):
            continue
            
        # Upper band logic
        if i == 10:
            final_upper[i] = upper_band[i]
            final_lower[i] = lower_band[i]
        else:
            if close_1d[i-1] <= final_upper[i-1]:
                final_upper[i] = upper_band[i]
            else:
                final_upper[i] = max(upper_band[i], final_upper[i-1])
                
            if close_1d[i-1] >= final_lower[i-1]:
                final_lower[i] = lower_band[i]
            else:
                final_lower[i] = min(lower_band[i], final_lower[i-1])
        
        # Trend direction
        if i == 10:
            if close_1d[i] > final_upper[i-1]:
                direction[i] = 1
            else:
                direction[i] = -1
        else:
            if direction[i-1] == -1 and close_1d[i] > final_upper[i]:
                direction[i] = 1
            elif direction[i-1] == 1 and close_1d[i] < final_lower[i]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]
        
        # Supertrend value
        if direction[i] == 1:
            supertrend[i] = final_lower[i]
        else:
            supertrend[i] = final_upper[i]
    
    # Align Supertrend to 4h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    st_uptrend = direction == 1  # 1d Supertrend direction
    st_uptrend_aligned = align_htf_to_ltf(prices, df_1d, st_uptrend.astype(float))
    st_downtrend = direction == -1
    st_downtrend_aligned = align_htf_to_ltf(prices, df_1d, st_downtrend.astype(float))
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
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
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for Supertrend/ATR/Donchian)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(supertrend_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr_4h_raw[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_4h_raw[i]
        st_up = st_uptrend_aligned[i] == 1.0
        st_down = st_downtrend_aligned[i] == 1.0
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Donchian lower band
            if price < donchian_lower[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Donchian upper band
            if price > donchian_upper[i]:
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
            # LONG: Price breaks above Donchian upper AND Supertrend uptrend AND volume spike
            if price > donchian_upper[i] and st_up and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Donchian lower AND Supertrend downtrend AND volume spike
            elif price < donchian_lower[i] and st_down and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Donchian20_1dSupertrend10_3_VolumeSpike_V1"
timeframe = "4h"
leverage = 1.0