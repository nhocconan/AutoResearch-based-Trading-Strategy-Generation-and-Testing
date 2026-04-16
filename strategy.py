#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and ATR trailing stop.
# Long when price breaks above Donchian high AND 12h volume > 1.5x 30-period average.
# Short when price breaks below Donchian low AND 12h volume > 1.5x 30-period average.
# Exit on ATR trailing stop (3*ATR from extreme) or opposite Donchian break.
# Uses discrete position size 0.25. Volume confirmation and ATR trailing reduce whipsaws.
# Target: 75-150 total trades over 4 years (19-38/year) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 12h Indicators: Volume Spike (volume > 1.5x 30-period average) ===
    df_12h = get_htf_data(prices, '12h')
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=30, min_periods=30).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    volume_spike = volume > (1.5 * vol_ma_12h_aligned)
    
    # === 4h ATR for trailing stop ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h_raw = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 30 periods needed for 12h vol MA)
    warmup = 60
    
    # Track position state and extreme price for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0
    short_extreme = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(volume_spike[i]) or
            np.isnan(atr_4h_raw[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            long_extreme = 0.0
            short_extreme = 0.0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_4h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Update long extreme
            if price > long_extreme:
                long_extreme = price
            # Exit if price breaks below Donchian low (opposite breakout)
            if price < low_roll[i]:
                exit_signal = True
            # ATR trailing stop: 3*ATR below long extreme
            elif price < long_extreme - 3.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Update short extreme (lowest price)
            if short_extreme == 0.0 or price < short_extreme:
                short_extreme = price
            # Exit if price breaks above Donchian high (opposite breakout)
            if price > high_roll[i]:
                exit_signal = True
            # ATR trailing stop: 3*ATR above short extreme
            elif price > short_extreme + 3.0 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            long_extreme = 0.0
            short_extreme = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian high AND volume spike
            if price > high_roll[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                long_extreme = price
                short_extreme = 0.0
            
            # SHORT: Price breaks below Donchian low AND volume spike
            elif price < low_roll[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                short_extreme = price
                long_extreme = 0.0
        
        else:
            # Maintain position
            signals[i] = position * 0.25
    
    return signals

name = "4h_Donchian20_12hVolumeSpike_ATRTrail_V1"
timeframe = "4h"
leverage = 1.0