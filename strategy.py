#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above Donchian upper band AND price > 1d EMA34 AND 6h volume > 1.5x 20-period average.
# Short when price breaks below Donchian lower band AND price < 1d EMA34 AND 6h volume > 1.5x 20-period average.
# Exit on ATR-based trailing stop (2.5*ATR from extreme) or opposite Donchian breakout.
# Uses discrete position size 0.28. Works in both bull and bear markets by requiring
# volume confirmation and trend alignment via 1d EMA34. Target: 75-200 total trades over 4 years (19-50/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    donchian_upper_prev = np.roll(donchian_upper, 1)
    donchian_lower_prev = np.roll(donchian_lower, 1)
    donchian_upper_prev[0] = np.nan
    donchian_lower_prev[0] = np.nan
    
    # === 1d Indicators: EMA34 ===
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 6h ATR for trailing stop ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_6h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_6h_raw = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC (avoid low liquidity periods)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 34 periods needed for EMA34)
    warmup = 50
    
    # Track position state, entry price, and extreme price for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    extreme_price = 0.0  # tracks highest high for long, lowest low for short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_upper_prev[i]) or
            np.isnan(donchian_lower_prev[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_6h_raw[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        atr_val = atr_6h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Update extreme price (highest high)
            if high[i] > extreme_price:
                extreme_price = high[i]
            # ATR-based trailing stop: 2.5*ATR below extreme price
            if price < extreme_price - 2.5 * atr_val:
                exit_signal = True
            # Opposite Donchian breakout (break below lower band)
            elif price < donchian_lower[i] and donchian_lower_prev[i] >= donchian_lower_prev[i-1] if i > 0 else False:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Update extreme price (lowest low)
            if low[i] < extreme_price:
                extreme_price = low[i]
            # ATR-based trailing stop: 2.5*ATR above extreme price
            if price > extreme_price + 2.5 * atr_val:
                exit_signal = True
            # Opposite Donchian breakout (break above upper band)
            elif price > donchian_upper[i] and donchian_upper_prev[i] <= donchian_upper_prev[i-1] if i > 0 else False:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            extreme_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian upper band AND price > 1d EMA34 AND volume spike
            vol_spike = volume[i] > (1.5 * pd.Series(volume).rolling(window=20, min_periods=20).mean().iloc[i]) if not np.isnan(pd.Series(volume).rolling(window=20, min_periods=20).mean().iloc[i]) else False
            if (price > donchian_upper[i] and close[i-1] <= donchian_upper[i-1] and  # clean breakout above
                price > ema_34_1d_aligned[i] and vol_spike):
                signals[i] = 0.28
                position = 1
                entry_price = price
                extreme_price = price
            
            # SHORT: Price breaks below Donchian lower band AND price < 1d EMA34 AND volume spike
            elif (price < donchian_lower[i] and close[i-1] >= donchian_lower[i-1] and  # clean breakdown below
                  price < ema_34_1d_aligned[i] and vol_spike):
                signals[i] = -0.28
                position = -1
                entry_price = price
                extreme_price = price
        
        else:
            signals[i] = position * 0.28
    
    return signals

name = "6h_Donchian20_1dEMA34_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0