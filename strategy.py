#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian(20) breakout with volume confirmation and ATR-based stoploss.
# Long when price breaks above 1w Donchian(20) upper band AND volume > 1.5x 20-day average.
# Short when price breaks below 1w Donchian(20) lower band AND volume > 1.5x 20-day average.
# Exit when price crosses the 1w Donchian middle band (20-period average) OR ATR stoploss triggered.
# Uses discrete position size 0.25. 1w filter provides signal direction, 1d provides entry timing and risk management.
# Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag.
# Works in both bull and bear markets: breakouts capture strong trends, volume confirmation filters false signals,
# ATR stoploss limits drawdowns during reversals/choppy periods.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data once before loop for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === 1w Indicators: Donchian Channels (20) ===
    donchian_upper_20_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_lower_20_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_middle_20_1w = (donchian_upper_20_1w + donchian_lower_20_1w) / 2.0
    
    # === 1d Indicators: ATR (14) for stoploss ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's ATR (using EWM with alpha=1/14)
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align all indicators to primary timeframe (1d)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper_20_1w)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower_20_1w)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1w, donchian_middle_20_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50
    
    # Track position state and entry price for stoploss
    position = 0
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values (aligned)
        donchian_upper = donchian_upper_aligned[i]
        donchian_lower = donchian_lower_aligned[i]
        donchian_middle = donchian_middle_aligned[i]
        price = close[i]
        vol = volume[i]
        atr = atr_14[i]
        
        # Get 1d volume average
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below middle band OR ATR stoploss hit (2*ATR below entry)
            if price < donchian_middle or price < entry_price - 2.0 * atr:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above middle band OR ATR stoploss hit (2*ATR above entry)
            if price > donchian_middle or price > entry_price + 2.0 * atr:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian upper AND volume > 1.5x 20-day average
            if price > donchian_upper and vol > 1.5 * vol_ma_20:
                signals[i] = 0.25
                position = 1
                entry_price = price  # Approximate entry price for stoploss calculation
            
            # SHORT: Price breaks below Donchian lower AND volume > 1.5x 20-day average
            elif price < donchian_lower and vol > 1.5 * vol_ma_20:
                signals[i] = -0.25
                position = -1
                entry_price = price  # Approximate entry price for stoploss calculation
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_1wDonchian20_VolumeConfirmation_ATRStop_V1"
timeframe = "1d"
leverage = 1.0