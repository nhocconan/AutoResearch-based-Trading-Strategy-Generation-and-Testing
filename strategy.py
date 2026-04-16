#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1w EMA(200) trend filter and volume confirmation.
# Long when price breaks above Donchian upper band AND 1w EMA(200) trending up AND volume > 1.5x 20-period average.
# Short when price breaks below Donchian lower band AND 1w EMA(200) trending down AND volume > 1.5x 20-period average.
# Uses ATR-based trailing stop: signal → 0 when price < highest_high_since_entry - 3*ATR (long) or price > lowest_low_since_entry + 3*ATR (short).
# Weekly EMA(200) provides strong trend filter that works in both bull and bear markets (avoids counter-trend trades).
# Volume confirmation ensures breakouts have conviction.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while capturing strong momentum moves.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Donchian Channel (20) ===
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h ATR for volatility ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_6h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_6h_raw = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # === 1w Indicators: EMA(200) for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    ema_200_up = ema_200_1w_aligned > np.roll(ema_200_1w_aligned, 1)
    ema_200_down = ema_200_1w_aligned < np.roll(ema_200_1w_aligned, 1)
    
    # === 6h Volume Confirmation: volume > 1.5x 20-period average ===
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_6h)
    
    # Session filter: 08-20 UTC (avoid low-liquidity Asian session)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 200  # EMA(200) needs 200 periods
    
    # Track position state and extremes for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(atr_6h_raw[i]) or
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(volume_spike[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        atr_val = atr_6h_raw[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Update highest high since entry
            if price > highest_since_entry:
                highest_since_entry = price
            # Trailing stop: exit if price drops 3*ATR from highest since entry
            elif price < highest_since_entry - 3.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if price < lowest_since_entry:
                lowest_since_entry = price
            # Trailing stop: exit if price rises 3*ATR from lowest since entry
            elif price > lowest_since_entry + 3.0 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian upper AND weekly EMA trending up AND volume spike
            if price > donchian_upper[i] and ema_200_up[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
                lowest_since_entry = price
            
            # SHORT: Price breaks below Donchian lower AND weekly EMA trending down AND volume spike
            elif price < donchian_lower[i] and ema_200_down[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
                highest_since_entry = price
                lowest_since_entry = price
        
        else:
            # Hold current position
            signals[i] = position * 0.25
    
    return signals

name = "6h_Donchian20_1wEMA200_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0