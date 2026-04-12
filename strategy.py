# 12h_1w_camarilla_breakout_volume_regime
# Hypothesis: 12-hour chart with weekly Camarilla levels (from weekly candles) plus volume confirmation and chop regime filter. Weekly timeframe reduces noise, volume confirms breakout strength, chop filter avoids range-bound whipsaws. Designed for low trade frequency (<30/year) to minimize fee drag while capturing strong trending moves in both bull and bear markets.

name = "12h_1w_camarilla_breakout_volume_regime"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for higher timeframe context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's range for Camarilla levels
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    
    # Calculate weekly ATR for chop filter (14-period)
    tr1 = np.abs(np.subtract(high_1w, low_1w))
    tr2 = np.abs(np.subtract(high_1w, np.roll(close_1w, 1)))
    tr3 = np.abs(np.subtract(low_1w, np.roll(close_1w, 1)))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate True Range for chop calculation
    true_range = tr
    
    # Choppiness Index calculation (14-period)
    # CHOP = 100 * log10(sum(TR14) / (ATR14 * 14)) / log10(14)
    atr_sum = pd.Series(true_range).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(atr_sum / (atr_1w * 14)) / np.log10(14)
    
    # Previous week's Camarilla levels (based on previous weekly candle)
    range_ = prev_high - prev_low
    # Resistance levels
    r3 = prev_close + range_ * 1.1 / 2
    r4 = prev_close + range_ * 1.1
    # Support levels
    s3 = prev_close - range_ * 1.1 / 2
    s4 = prev_close - range_ * 1.1
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.8)
    
    # Align all weekly indicators to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    vol_confirm_aligned = align_htf_to_ltf(prices, df_1w, vol_confirm.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(vol_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Chop regime filter: only trade when market is trending (CHOP < 40)
        is_trending = chop_aligned[i] < 40
        
        # Long entry: close breaks above R4 with volume confirmation and trending market
        if (close[i] > r4_aligned[i] and vol_confirm_aligned[i] > 0.5 and 
            is_trending and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: close breaks below S4 with volume confirmation and trending market
        elif (close[i] < s4_aligned[i] and vol_confirm_aligned[i] > 0.5 and 
              is_trending and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or close crosses back to opposite S3/R3
        elif position == 1 and close[i] < s3_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > r3_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals