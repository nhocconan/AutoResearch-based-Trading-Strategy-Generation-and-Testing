#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA200 trend filter and volume confirmation.
# Long when price breaks above 20-period high AND price > 1w EMA200 AND 12h volume > 1.8x 20-period average.
# Short when price breaks below 20-period low AND price < 1w EMA200 AND 12h volume > 1.8x 20-period average.
# Exit on ATR-based stoploss (2.5*ATR from entry) or opposite breakout.
# Uses discrete position size 0.30. Designed for 12h timeframe with low trade frequency (target: 50-150/4 years).
# Works in both bull and bear markets by requiring volume confirmation and strong trend alignment via 1w EMA200.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: Donchian Channel (20-period) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1w Indicators: EMA200 for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    ema_200_1w = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # === 12h Volume Spike Confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # === 12h ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h_raw = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC (avoid low-volume Asian session)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 200 periods needed for EMA200)
    warmup = 220
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema_200_1w_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr_12h_raw[i]) or not session_filter[i]):
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
            # Exit if price breaks below 12-period low (shorter-term mean reversion)
            if price < lowest_low[i]:
                exit_signal = True
            # ATR-based stoploss: 2.5*ATR below entry
            elif price < entry_price - 2.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above 12-period high (shorter-term mean reversion)
            if price > highest_high[i]:
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
        if position == 0:
            # LONG: Price breaks above 20-period high AND price > 1w EMA200 AND volume spike
            if (price > highest_high[i] and 
                price > ema_200_1w_aligned[i] and vol_spike):
                signals[i] = 0.30
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below 20-period low AND price < 1w EMA200 AND volume spike
            elif (price < lowest_low[i] and 
                  price < ema_200_1w_aligned[i] and vol_spike):
                signals[i] = -0.30
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.30
    
    return signals

name = "12h_Donchian20_1wEMA200_VolumeSpike_V1"
timeframe = "12h"
leverage = 1.0