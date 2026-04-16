#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout direction + 1h volume spike + session filter (08-20 UTC).
# Long when 4h price > 4h Donchian(20) high AND 1h volume > 2.0x 20-period average AND in session.
# Short when 4h price < 4h Donchian(20) low AND 1h volume > 2.0x 20-period average AND in session.
# Exit on opposite 4h Donchian break or ATR-based stop (1.5*ATR from entry).
# Uses discrete position size 0.20. Target: 60-150 total trades over 4 years (15-37/year).
# Works in bull/bear by requiring volume confirmation and symmetric breakout levels.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    high_roll_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_roll_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_4h_aligned = align_htf_to_ltf(prices, df_4h, high_roll_4h)
    donchian_low_4h_aligned = align_htf_to_ltf(prices, df_4h, low_roll_4h)
    
    # === 1h Indicators: Volume Spike (volume > 2.0x 20-period average) ===
    vol_ma_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_1h)
    
    # === 1h ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_1h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1h_raw = pd.Series(tr_1h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods needed)
    warmup = 50
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(donchian_high_4h_aligned[i]) or np.isnan(donchian_low_4h_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr_1h_raw[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_1h_raw[i]
        donchian_high = donchian_high_4h_aligned[i]
        donchian_low = donchian_low_4h_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below 4h Donchian low (opposite breakout)
            if price < donchian_low:
                exit_signal = True
            # ATR-based stoploss: 1.5*ATR below entry
            elif price < entry_price - 1.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above 4h Donchian high (opposite breakout)
            if price > donchian_high:
                exit_signal = True
            # ATR-based stoploss: 1.5*ATR above entry
            elif price > entry_price + 1.5 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price above 4h Donchian high AND volume spike AND in session
            if price > donchian_high and vol_spike:
                signals[i] = 0.20
                position = 1
                entry_price = price
            
            # SHORT: Price below 4h Donchian low AND volume spike AND in session
            elif price < donchian_low and vol_spike:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_Donchian4h_1hVolSpike_Session_ATRStop_V1"
timeframe = "1h"
leverage = 1.0