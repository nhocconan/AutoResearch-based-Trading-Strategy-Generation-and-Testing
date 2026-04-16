#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND price > 1d EMA50 (uptrend) AND 4h volume > 1.5x 20-period average.
# Short when price breaks below Donchian(20) low AND price < 1d EMA50 (downtrend) AND 4h volume > 1.5x 20-period average.
# Uses discrete position size 0.25. Donchian provides structure, 1d EMA50 ensures higher timeframe alignment,
# volume spike confirms breakout validity. Designed for 4h timeframe to limit trades (target: 80-160 over 4 years).
# ATR-based stoploss implemented via signal=0 when price moves against position by 2.0*ATR(14).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get 1d data once before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA50 calculation
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: EMA50 for trend filter ===
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for EMA, 20 for Donchian/volume)
    warmup = 60
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        price = close[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        ema_1d = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        # Update trailing stoploss for existing positions
        if position == 1:  # Long position
            # Stoploss: price < entry_price - 2.0 * atr
            if price < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Stoploss: price > entry_price + 2.0 * atr
            if price > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC (channel reversal) ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price falls below Donchian lower channel
            if price < lower_channel:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price rises above Donchian upper channel
            if price > upper_channel:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price breaks above Donchian upper channel AND price > 1d EMA50 AND volume spike
            if price > upper_channel and price > ema_1d and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: price breaks below Donchian lower channel AND price < 1d EMA50 AND volume spike
            elif price < lower_channel and price < ema_1d and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Donchian20_1dEMA50_VolumeSpike_ATRStop_V1"
timeframe = "4h"
leverage = 1.0