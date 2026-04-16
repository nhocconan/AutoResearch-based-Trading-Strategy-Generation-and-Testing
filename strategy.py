#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA(21) trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND price > 12h EMA(21) AND volume > 1.5x 20-period 4h average volume.
# Short when price breaks below Donchian(20) low AND price < 12h EMA(21) AND volume > 1.5x 20-period 4h average volume.
# Exit when price reverts to Donchian(20) midpoint or ATR-based stoploss (2*ATR from entry).
# Uses discrete position size 0.25. Designed to capture breakouts in trending markets with volume confirmation.
# Works in both bull and bear markets by requiring trend (price vs 12h EMA) and volume confirmation, avoiding false breakouts.
# Target: 75-200 total trades over 4 years (19-50/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # === 4h Indicators: ATR (14-period) for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # === 4h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_4h)
    
    # === 12h Indicators: EMA(21) for trend filter ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for ATR/EMA)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(atr_4h[i]) or np.isnan(volume_spike[i]) or np.isnan(ema_21_12h_aligned[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        dc_high = donchian_high[i]
        dc_low = donchian_low[i]
        dc_mid = donchian_mid[i]
        vol_spike = volume_spike[i]
        ema_trend = ema_21_12h_aligned[i]
        atr_val = atr_4h[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price reverts to Donchian midpoint
            if price <= dc_mid:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price reverts to Donchian midpoint
            if price >= dc_mid:
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
            # LONG: Price breaks above Donchian high AND price > 12h EMA(21) AND volume spike
            if price > dc_high and price > ema_trend and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Donchian low AND price < 12h EMA(21) AND volume spike
            elif price < dc_low and price < ema_trend and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Donchian20_12hEMA21_VolumeSpike_V1"
timeframe = "4h"
leverage = 1.0