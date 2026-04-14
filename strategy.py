#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray Index (Bull/Bear Power) with 1-day regime filter
# Bull Power = High - EMA13, Bear Power = EMA13 - Low
# Long when Bull Power > 0 AND 1d close > 1d EMA50 (bullish regime)
# Short when Bear Power > 0 AND 1d close < 1d EMA50 (bearish regime)
# Exit when power crosses zero
# Elder Ray measures bull/bear strength relative to trend (EMA), regime filter ensures alignment with higher timeframe trend
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while capturing trends

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 6h and 1d data ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Elder Ray on 6h: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    close_6h = df_6h['close'].values
    ema13 = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Calculate 1d EMA50 for regime filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get 1d regime: bullish if close > EMA50, bearish if close < EMA50
        regime_bullish = close_1d_aligned[i] > ema50_1d_aligned[i] if 'close_1d_aligned' in locals() else True
        # Actually need to get aligned close_1d
        if 'close_1d_aligned' not in locals():
            close_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
            regime_bullish = close_1d_aligned[i] > ema50_1d_aligned[i]
        
        if position == 0:
            # Long setup: Bull Power > 0 AND bullish regime (1d close > EMA50)
            if bull_power_aligned[i] > 0 and regime_bullish:
                position = 1
                signals[i] = position_size
            # Short setup: Bear Power > 0 AND bearish regime (1d close < EMA50)
            elif bear_power_aligned[i] > 0 and not regime_bullish:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bull Power crosses below 0
            if bull_power_aligned[i] <= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Bear Power crosses below 0
            if bear_power_aligned[i] <= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_ElderRay_1dRegime"
timeframe = "6h"
leverage = 1.0