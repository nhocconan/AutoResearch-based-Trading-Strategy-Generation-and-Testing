#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and 1d volume spike confirmation.
# Long when price breaks above Donchian upper channel AND 1w EMA50 rising AND 1d volume > 1.5x 20-period average.
# Short when price breaks below Donchian lower channel AND 1w EMA50 falling AND 1d volume > 1.5x 20-period average.
# Exit on opposite Donchian channel touch or ATR-based stoploss (2.5*ATR from entry).
# Uses discrete position size 0.25. Designed to capture strong trends with volume confirmation in both bull and bear markets.
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: Donchian Channel (20-period) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 1w Indicators: EMA50 Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    ema_50_1w_rising = ema_50_1w_aligned > np.roll(ema_50_1w_aligned, 1)
    ema_50_1w_falling = ema_50_1w_aligned < np.roll(ema_50_1w_aligned, 1)
    # Handle first value
    ema_50_1w_rising[0] = False
    ema_50_1w_falling[0] = False
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # === 12h ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for EMA50/ATR/Donchian)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr_12h[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        ema_rising = ema_50_1w_rising[i]
        ema_falling = ema_50_1w_falling[i]
        atr_val = atr_12h[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price touches or crosses below Donchian lower channel
            if price <= lowest_low[i]:
                exit_signal = True
            # ATR-based stoploss: 2.5*ATR below entry
            elif price < entry_price - 2.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price touches or crosses above Donchian upper channel
            if price >= highest_high[i]:
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
            # LONG: Price breaks above Donchian upper channel AND 1w EMA50 rising AND volume spike
            if price > highest_high[i] and ema_rising and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Donchian lower channel AND 1w EMA50 falling AND volume spike
            elif price < lowest_low[i] and ema_falling and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_Donchian20_1wEMA50_1dVolumeSpike_V1"
timeframe = "12h"
leverage = 1.0