#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# - Williams Alligator: Jaw (13-period SMMA, 8-bar offset), Teeth (8-period SMMA, 5-bar offset), Lips (5-period SMMA, 3-bar offset)
# - Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA200 AND volume > 1.5x 20-bar avg
# - Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA200 AND volume > 1.5x 20-bar avg
# - Exit when Alligator lines converge (|Lips - Jaw| < 0.1 * ATR(14)) indicating loss of momentum
# - Uses 1d EMA200 for higher timeframe trend filter to avoid counter-trend trades
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-30 trades/year on 12h timeframe (50-120 total over 4 years)

name = "12h_1d_alligator_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Pre-compute 1d EMA(200) for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Pre-compute Williams Alligator on 12h data
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 100:
        return np.zeros(n)
    
    median_12h = (df_12h['high'] + df_12h['low']) / 2
    close_12h = df_12h['close'].values
    median_12h_vals = median_12h.values
    
    # SMMA (Smoothed Moving Average) calculation
    def smma(data, period):
        if len(data) < period:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        sma = np.mean(data[:period])
        result[period-1] = sma
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Alligator lines: Jaw (13,8), Teeth (8,5), Lips (5,3)
    jaw_raw = smma(median_12h_vals, 13)
    teeth_raw = smma(median_12h_vals, 8)
    lips_raw = smma(median_12h_vals, 5)
    
    # Apply offsets: Jaw +8, Teeth +5, Lips +3
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Invalidate the rolled values that don't have enough data
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Align Alligator lines to 12h timeframe (no additional delay needed for SMMA)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Pre-compute ATR(14) for exit condition
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new Alligator alignment entries
            # Bullish alignment: Lips > Teeth > Jaw
            bullish = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
            # Bearish alignment: Lips < Teeth < Jaw
            bearish = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
            
            # Long when bullish alignment AND price > 1d EMA200 AND volume spike
            if bullish and (prices['close'].iloc[i] > ema200_1d_aligned[i]) and vol_spike.iloc[i]:
                position = 1
                signals[i] = 0.25
            # Short when bearish alignment AND price < 1d EMA200 AND volume spike
            elif bearish and (prices['close'].iloc[i] < ema200_1d_aligned[i]) and vol_spike.iloc[i]:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit when Alligator converges
            # Exit when Alligator lines converge (loss of momentum)
            # Convergence: |Lips - Jaw| < 0.1 * ATR(14)
            convergence = np.abs(lips_aligned[i] - jaw_aligned[i]) < (0.1 * atr[i])
            
            if convergence:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals