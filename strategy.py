#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + Williams Fractal confluence with 1w trend filter
# - Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
# - Williams Fractal: Bearish = High[2] < High[0] and High[2] < High[1] and High[0] < High[1] and High[0] < High[3] and High[0] < High[4]
#                  Bullish = Low[2] > Low[0] and Low[2] > Low[1] and Low[0] < Low[1] and Low[0] > Low[3] and Low[0] > Low[4]
# - Long when Bull Power > 0 AND Bearish Fractal (sell signal) AND price > 1w EMA50
# - Short when Bear Power > 0 AND Bullish Fractal (buy signal) AND price < 1w EMA50
# - Exit when Elder Ray power crosses zero (Bull Power <= 0 for longs, Bear Power <= 0 for shorts)
# - Uses 1w EMA50 for trend filter to avoid counter-trend trades
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-30 trades/year on 6h timeframe (48-120 total over 4 years)
# - Elder Ray measures bull/bear power behind moves; Williams Fractal provides timing
# - Works in both bull (trend continuation) and bear (counter-trend bounces) markets

name = "6h_1w_elder_ray_fractal_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute Elder Ray (13-period EMA)
    close_s = pd.Series(prices['close'])
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = prices['high'].values - ema13
    bear_power = ema13 - prices['low'].values
    
    # Pre-compute Williams Fractals on 6h data
    high = prices['high'].values
    low = prices['low'].values
    bearish_fractal = np.zeros(n, dtype=bool)
    bullish_fractal = np.zeros(n, dtype=bool)
    # Williams Fractal: need 2 bars on each side
    for i in range(2, n-2):
        # Bearish fractal: highest high in middle with lower highs on both sides
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            bearish_fractal[i] = True
        # Bullish fractal: lowest low in middle with higher lows on both sides
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            bullish_fractal[i] = True
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute aligned 1w data
    c_1w = df_1w['close'].values
    c_1w_aligned = align_htf_to_ltf(prices, df_1w, c_1w)
    
    # Pre-compute 1w EMA(50) for trend filter
    ema50_1w = pd.Series(c_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    for i in range(50, n):  # Start after EMA50 warmup
        # Skip if any required data is invalid
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(c_1w_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long when Bull Power > 0 AND Bearish Fractal (sell signal = potential top) with volume spike AND in 1w uptrend
            if (bull_power[i] > 0 and 
                bearish_fractal[i] and 
                vol_spike.iloc[i] and
                prices['close'].iloc[i] > ema50_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short when Bear Power > 0 AND Bullish Fractal (buy signal = potential bottom) with volume spike AND in 1w downtrend
            elif (bear_power[i] > 0 and 
                  bullish_fractal[i] and 
                  vol_spike.iloc[i] and
                  prices['close'].iloc[i] < ema50_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when Elder Ray power crosses zero (trend weakness)
            exit_signal = False
            if position == 1:  # Long position
                if bull_power[i] <= 0:
                    exit_signal = True
            elif position == -1:  # Short position
                if bear_power[i] <= 0:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals