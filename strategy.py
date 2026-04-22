#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with daily ATR filter and volume confirmation
# Long when price breaks above Donchian(20) high + ATR(14) rising + volume spike
# Short when price breaks below Donchian(20) low + ATR(14) rising + volume spike
# Exit when price returns to Donchian midline or ATR declines
# Designed for moderate trade frequency (~20-30/year) with strong trend-following edge
# Works in bull markets via breakouts and in bear markets via short breakdowns

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for ATR filter
    df_daily = get_htf_data(prices, '1d')
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate Donchian channels (20-period) on 4h data
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Calculate daily ATR(14) for trend strength filter
    tr1 = high_daily[1:] - low_daily[1:]
    tr2 = np.abs(high_daily[1:] - close_daily[:-1])
    tr3 = np.abs(low_daily[1:] - close_daily[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_rising = atr_14 > np.roll(atr_14, 1)  # ATR rising vs previous day
    
    # Align daily ATR and ATR rising to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_daily, atr_14)
    atr_rising_aligned = align_htf_to_ltf(prices, df_daily, atr_rising.astype(float))
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(donch_mid[i]) or 
            np.isnan(atr_14_aligned[i]) or 
            np.isnan(atr_rising_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = donch_high[i]
        lower = donch_low[i]
        midline = donch_mid[i]
        atr_val = atr_14_aligned[i]
        atr_rise = atr_rising_aligned[i] > 0.5  # Convert back to boolean
        
        # Volume filter: current volume > 1.8 * 20-period average
        vol_spike = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above Donchian high + ATR rising + volume spike
            if price > upper and atr_rise and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low + ATR rising + volume spike
            elif price < lower and atr_rise and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price returns to midline or ATR declines
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price returns to midline or ATR stops rising
                if price <= midline or not atr_rise:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price returns to midline or ATR stops rising
                if price >= midline or not atr_rise:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_Breakout_DailyATR_Volume"
timeframe = "4h"
leverage = 1.0