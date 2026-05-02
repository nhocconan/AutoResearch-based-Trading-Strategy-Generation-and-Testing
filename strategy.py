#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + Williams Alligator with 1d trend filter
# Elder Ray measures bull/bear power (close-EMA13) to detect momentum strength
# Williams Alligator (jaw/teeth/lips) identifies trend and avoids choppy markets
# 1d EMA50 provides higher timeframe bias to trade with the dominant trend
# Volume confirmation ensures strong participation on signals
# Designed for 6h timeframe to target 50-150 total trades over 4 years (12-37/year)
# Works in bull markets via long signals when bull power > 0 and price above Alligator
# Works in bear markets via short signals when bear power < 0 and price below Alligator
# Filters out ranging markets where Alligator lines are intertwined

name = "6h_ElderRay_Alligator_1dEMA50_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate EMA13 for Elder Ray (primary timeframe)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) - all SMMA
    def smma(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        sma = np.mean(data[:period])
        result[period-1] = sma
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line
    lips = smma(close, 5)   # Green line
    
    # Volume confirmation (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Alligator)
    start_idx = 35
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment check: teeth > lips > jaw for uptrend, jaw > teeth > lips for downtrend
        alligator_long = teeth[i] > lips[i] and lips[i] > jaw[i]
        alligator_short = jaw[i] > teeth[i] and teeth[i] > lips[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull power > 0 + price above Alligator (teeth) + volume spike + price > 1d EMA50
            if bull_power[i] > 0 and close[i] > teeth[i] and alligator_long and volume_spike[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear power < 0 + price below Alligator (teeth) + volume spike + price < 1d EMA50
            elif bear_power[i] < 0 and close[i] < teeth[i] and alligator_short and volume_spike[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bear power < 0 OR price below jaw OR price < 1d EMA50
            if bear_power[i] < 0 or close[i] < jaw[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bull power > 0 OR price above jaw OR price > 1d EMA50
            if bull_power[i] > 0 or close[i] > jaw[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals