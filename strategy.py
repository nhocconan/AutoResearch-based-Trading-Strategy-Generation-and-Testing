#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1w Williams Fractal (bearish) for HTF resistance + 6h Donchian(20) breakout with volume confirmation.
# Short when price < 1w bearish fractal AND breaks below 6h Donchian lower(20) with volume spike (>1.8x median volume).
# Long when price > 1w bullish fractal AND breaks above 6h Donchian upper(20) with volume spike.
# Exit via ATR(10) trailing stop: long exits when price < highest high since entry - 2.5*ATR,
# short exits when price > lowest low since entry + 2.5*ATR.
# Uses discrete position size 0.25. Volume spike and ATR trailing stop reduce whipsaw and overtrading.
# Williams Fractal identifies key swing points from weekly structure, effective in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data once before loop for Williams Fractal
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # === 1w Indicators: Williams Fractal ===
    # Bearish fractal: high[n-2] < high[n-1] and high[n] < high[n-1] and high[n+1] < high[n-1] and high[n+2] < high[n-1]
    # Bullish fractal: low[n-2] > low[n-1] and low[n] > low[n-1] and low[n+1] > low[n-1] and low[n+2] > low[n-1]
    bearish_fractal = np.full(len(high_1w), np.nan)
    bullish_fractal = np.full(len(low_1w), np.nan)
    
    for i in range(2, len(high_1w) - 2):
        if (high_1w[i-2] < high_1w[i-1] and 
            high_1w[i] < high_1w[i-1] and 
            high_1w[i+1] < high_1w[i-1] and 
            high_1w[i+2] < high_1w[i-1]):
            bearish_fractal[i] = high_1w[i-1]
        
        if (low_1w[i-2] > low_1w[i-1] and 
            low_1w[i] > low_1w[i-1] and 
            low_1w[i+1] > low_1w[i-1] and 
            low_1w[i+2] > low_1w[i-1]):
            bullish_fractal[i] = low_1w[i-1]
    
    # Forward fill to last valid fractal level
    bearish_fractal = pd.Series(bearish_fractal).ffill().bfill().values
    bullish_fractal = pd.Series(bullish_fractal).ffill().bfill().values
    
    # Get 6h data for Donchian, volume, and ATR
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # === 6h Indicators: Donchian(20), Volume Median, ATR(10) ===
    # Donchian channels (20-period)
    donchian_upper_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_lower_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Volume median (20-period)
    vol_median_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).median().values
    
    # ATR(10) for trailing stop
    tr1_6h = pd.Series(high_6h - low_6h)
    tr2_6h = pd.Series(np.abs(high_6h - np.roll(close_6h, 1)))
    tr3_6h = pd.Series(np.abs(low_6h - np.roll(close_6h, 1)))
    tr2_6h.iloc[0] = tr1_6h.iloc[0]
    tr3_6h.iloc[0] = tr1_6h.iloc[0]
    tr_6h = pd.concat([tr1_6h, tr2_6h, tr3_6h], axis=1).max(axis=1)
    atr_10 = pd.Series(tr_6h).rolling(window=10, min_periods=10).mean().values
    
    # Align all indicators to primary timeframe (6h)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractar, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_6h, donchian_upper_20)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_6h, donchian_lower_20)
    vol_median_aligned = align_htf_to_ltf(prices, df_6h, vol_median_20)
    atr_aligned = align_htf_to_ltf(prices, df_6h, atr_10)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(2, 20, 10)  # Williams Fractal lookback, Donchian(20), ATR(10)
    
    # Track position state for trailing stops
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_median_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
            continue
        
        # Current values (aligned)
        bearish_fractal_val = bearish_fractal_aligned[i]
        bullish_fractal_val = bullish_fractal_aligned[i]
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        vol_median = vol_median_aligned[i]
        atr = atr_aligned[i]
        price = close[i]
        
        # Get current 6h volume for volume spike filter
        vol_6h_aligned = align_htf_to_ltf(prices, df_6h, volume_6h)
        current_vol_6h = vol_6h_aligned[i]
        
        # Volume spike filter: current 6h volume > 1.8x median volume
        volume_spike = current_vol_6h > (vol_median * 1.8)
        
        # === EXIT LOGIC (trailing stop) ===
        exit_signal = False
        if position == 1:  # long position
            # Update highest high since entry
            if price > highest_since_entry:
                highest_since_entry = price
            # Exit when price drops below highest high - 2.5*ATR
            if price < highest_since_entry - 2.5 * atr:
                exit_signal = True
        elif position == -1:  # short position
            # Update lowest low since entry
            if price < lowest_since_entry:
                lowest_since_entry = price
            # Exit when price rises above lowest low + 2.5*ATR
            if price > lowest_since_entry + 2.5 * atr:
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
            # Price above 1w bullish fractal (support), breakout above Donchian upper, volume spike
            if price > bullish_fractal_val and price > upper and volume_spike:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price  # initialize trailing stop
            
            # SHORT CONDITIONS
            # Price below 1w bearish fractal (resistance), breakout below Donchian lower, volume spike
            elif price < bearish_fractal_val and price < lower and volume_spike:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price  # initialize trailing stop
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "6h_1wWilliamsFractal_6hDonchian20_Breakout_VolumeSpike1.8x_ATRTrail2.5_v1"
timeframe = "6h"
leverage = 1.0