#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d EMA(34) as trend filter and 4h Donchian(20) breakout with volume confirmation.
# Long when 1d EMA(34) is rising (trend up) AND price breaks above 4h Donchian upper band with volume spike (>1.5x median volume).
# Short when 1d EMA(34) is falling (trend down) AND price breaks below 4h Donchian lower band with volume spike.
# Exit via ATR(10) trailing stop: long exits when price < highest high since entry - 2.0*ATR,
# short exits when price > lowest low since entry + 2.0*ATR.
# Uses discrete position size 0.25. 1d EMA(34) identifies primary trend direction to avoid counter-trend trades,
# Donchian breakout captures momentum in the direction of the trend, volume confirmation reduces false signals.
# Designed to work in both bull and bear markets by following the primary trend with precise entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data once before loop for EMA(34) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: EMA(34) trend (rising/falling) ===
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Trend: rising if current EMA > previous EMA, falling if current EMA < previous EMA
    ema_34_1d_rising = np.zeros_like(ema_34_1d, dtype=bool)
    ema_34_1d_falling = np.zeros_like(ema_34_1d, dtype=bool)
    ema_34_1d_rising[1:] = ema_34_1d[1:] > ema_34_1d[:-1]
    ema_34_1d_falling[1:] = ema_34_1d[1:] < ema_34_1d[:-1]
    
    # Get 4h data for Donchian, volume, and ATR
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # === 4h Indicators: Donchian(20), Volume Median, ATR(10) ===
    # Donchian Channels (20-period)
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Volume median (20-period)
    vol_median_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).median().values
    
    # ATR(10) for trailing stop
    tr1_4h = pd.Series(high_4h - low_4h)
    tr2_4h = pd.Series(np.abs(high_4h - np.roll(close_4h, 1)))
    tr3_4h = pd.Series(np.abs(low_4h - np.roll(close_4h, 1)))
    tr2_4h.iloc[0] = tr1_4h.iloc[0]
    tr3_4h.iloc[0] = tr1_4h.iloc[0]
    tr_4h = pd.concat([tr1_4h, tr2_4h, tr3_4h], axis=1).max(axis=1)
    atr_10 = pd.Series(tr_4h).rolling(window=10, min_periods=10).mean().values
    
    # Align all indicators to primary timeframe (4h)
    ema_34_1d_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d_rising.astype(float))
    ema_34_1d_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d_falling.astype(float))
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    vol_median_aligned = align_htf_to_ltf(prices, df_4h, vol_median_20)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_10)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(34, 20, 10)  # EMA(34), Donchian(20), ATR(10)
    
    # Track position state for trailing stops
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_rising_aligned[i]) or np.isnan(ema_34_1d_falling_aligned[i]) or
            np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(vol_median_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
            continue
        
        # Current values (aligned)
        ema_rising = ema_34_1d_rising_aligned[i] > 0.5
        ema_falling = ema_34_1d_falling_aligned[i] > 0.5
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        vol_median = vol_median_aligned[i]
        atr = atr_aligned[i]
        price = close[i]
        
        # Get current 4h volume for volume spike filter
        vol_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_4h)
        current_vol_4h = vol_4h_aligned[i]
        
        # Volume spike filter: current 4h volume > 1.5x median volume
        volume_spike = current_vol_4h > (vol_median * 1.5)
        
        # === EXIT LOGIC (trailing stop) ===
        exit_signal = False
        if position == 1:  # long position
            # Update highest high since entry
            if price > highest_since_entry:
                highest_since_entry = price
            # Exit when price drops below highest high - 2.0*ATR
            if price < highest_since_entry - 2.0 * atr:
                exit_signal = True
        elif position == -1:  # short position
            # Update lowest low since entry
            if price < lowest_since_entry:
                lowest_since_entry = price
            # Exit when price rises above lowest low + 2.0*ATR
            if price > lowest_since_entry + 2.0 * atr:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG CONDITIONS
            # 1d EMA(34) rising, price breaks above Donchian upper, volume spike
            if ema_rising and price > upper and volume_spike:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price  # initialize trailing stop
            
            # SHORT CONDITIONS
            # 1d EMA(34) falling, price breaks below Donchian lower, volume spike
            elif ema_falling and price < lower and volume_spike:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price  # initialize trailing stop
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_1dEMA34_Trend_4hDonchian20_Breakout_VolumeSpike1.5x_ATRTrail2.0_v1"
timeframe = "4h"
leverage = 1.0