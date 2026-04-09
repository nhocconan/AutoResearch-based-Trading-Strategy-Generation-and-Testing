#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with volume confirmation and 12h ATR regime filter
# Uses Camarilla levels calculated from 12h OHLC: long at H4 breakout with volume > 2x avg, short at L4 breakdown
# Only takes trades when 12h ATR(14) is below its 50-period MA (low volatility regime where breakouts are more reliable)
# Works in both bull/bear: ATR regime filter avoids whipsaws in high volatility, volume confirmation ensures commitment
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag
# Position size 0.25 to manage drawdown and enable multiple concurrent positions

name = "6h_12h_camarilla_volume_atr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for Camarilla levels and ATR regime
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Calculate 12h ATR(14) for regime filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    tr_12h = np.full(len(df_12h), np.nan)
    for i in range(1, len(df_12h)):
        tr = max(
            high_12h[i] - low_12h[i],
            abs(high_12h[i] - close_12h[i-1]),
            abs(low_12h[i] - close_12h[i-1])
        )
        tr_12h[i] = tr
    
    atr_12h = np.full(len(df_12h), np.nan)
    for i in range(len(df_12h)):
        if i < 14:
            atr_12h[i] = np.nan
        elif i == 14:
            atr_12h[i] = np.nanmean(tr_12h[1:15])
        else:
            atr_12h[i] = (atr_12h[i-1] * 13 + tr_12h[i]) / 14
    
    # Calculate 50-period MA of ATR for regime filter
    atr_ma_50_12h = np.full(len(df_12h), np.nan)
    for i in range(len(df_12h)):
        if i < 50:
            atr_ma_50_12h[i] = np.nan
        else:
            atr_ma_50_12h[i] = np.mean(atr_12h[i-50:i])
    
    # Calculate Camarilla levels from 12h OHLC (based on previous day's range)
    camarilla_h4 = np.full(len(df_12h), np.nan)
    camarilla_l4 = np.full(len(df_12h), np.nan)
    camarilla_h3 = np.full(len(df_12h), np.nan)
    camarilla_l3 = np.full(len(df_12h), np.nan)
    
    for i in range(len(df_12h)):
        if i == 0:
            camarilla_h4[i] = np.nan
            camarilla_l4[i] = np.nan
            camarilla_h3[i] = np.nan
            camarilla_l3[i] = np.nan
        else:
            # Use previous 12h bar's OHLC to calculate current levels
            prev_close = close_12h[i-1]
            prev_range = high_12h[i-1] - low_12h[i-1]
            camarilla_h4[i] = prev_close + 1.5 * prev_range
            camarilla_l4[i] = prev_close - 1.5 * prev_range
            camarilla_h3[i] = prev_close + 1.25 * prev_range
            camarilla_l3[i] = prev_close - 1.25 * prev_range
    
    # Align 12h indicators to 6h timeframe
    camarilla_h4_6h = align_htf_to_ltf(prices, df_12h, camarilla_h4)
    camarilla_l4_6h = align_htf_to_ltf(prices, df_12h, camarilla_l4)
    camarilla_h3_6h = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    camarilla_l3_6h = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    atr_ma_50_6h = align_htf_to_ltf(prices, df_12h, atr_ma_50_12h)
    atr_6h = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4_6h[i]) or 
            np.isnan(camarilla_l4_6h[i]) or 
            np.isnan(atr_ma_50_6h[i]) or 
            np.isnan(atr_6h[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirm = volume[i] > 2.0 * avg_volume[i]
        
        # ATR regime filter: only trade when current ATR < ATR MA (low volatility regime)
        atr_regime = atr_6h[i] < atr_ma_50_6h[i]
        
        if position == 1:  # Long position
            # Exit conditions: price closes below Camarilla L3 OR ATR regime turns unfavorable
            if close[i] < camarilla_l3_6h[i] or not atr_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price closes above Camarilla H3 OR ATR regime turns unfavorable
            if close[i] > camarilla_h3_6h[i] or not atr_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Camarilla breakout with volume confirmation and ATR regime filter
            if volume_confirm and atr_regime:
                # Long breakout: price closes above Camarilla H4
                if close[i] > camarilla_h4_6h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price closes below Camarilla L4
                elif close[i] < camarilla_l4_6h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals