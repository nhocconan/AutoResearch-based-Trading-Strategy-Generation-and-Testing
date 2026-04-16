#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and 1d volume spike confirmation.
# Long when price breaks above 20-period 1d Donchian high AND 1w EMA50 is rising AND 1d volume > 1.5x 20-period average.
# Short when price breaks below 20-period 1d Donchian low AND 1w EMA50 is falling AND 1d volume > 1.5x 20-period average.
# Exit when price crosses the 10-period 1d EMA (dynamic stop/reversal).
# Uses discrete position size 0.25. Designed to capture major breakouts in strong trending markets.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag while maintaining edge.
# Works in both bull and bear markets by requiring EMA50 trend alignment and volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Indicators: Donchian Channel (20) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian high/low: 20-period rolling max/min
    donchian_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 1d timeframe
    donchian_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_1d)
    donchian_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_1d)
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # === 1d Indicators: 10-period EMA for exit ===
    ema_10_1d = pd.Series(close_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_10_1d)
    
    # === 1w Indicators: EMA50 trend (rising/falling) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    # Trend: rising if current > previous, falling if current < previous
    ema_50_1w_rising = ema_50_1w_aligned > np.roll(ema_50_1w_aligned, 1)
    ema_50_1w_falling = ema_50_1w_aligned < np.roll(ema_50_1w_aligned, 1)
    # Handle first value
    ema_50_1w_rising[0] = False
    ema_50_1w_falling[0] = False
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for EMA50)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(donchian_high_1d_aligned[i]) or np.isnan(donchian_low_1d_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(ema_10_1d_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        ema_10_val = ema_10_1d_aligned[i]
        is_ema50_rising = ema_50_1w_rising[i]
        is_ema50_falling = ema_50_1w_falling[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below 10-period EMA
            if price < ema_10_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above 10-period EMA
            if price > ema_10_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian high AND EMA50 rising AND volume spike
            if price > donchian_high_1d_aligned[i] and is_ema50_rising and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Donchian low AND EMA50 falling AND volume spike
            elif price < donchian_low_1d_aligned[i] and is_ema50_falling and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_1dVolumeSpike_V1"
timeframe = "1d"
leverage = 1.0