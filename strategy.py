#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# - Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend direction and strength
# - Long when Lips > Teeth > Jaw (bullish alignment) AND 1d EMA50 rising AND volume > 1.8x 20-bar avg
# - Short when Lips < Teeth < Jaw (bearish alignment) AND 1d EMA50 falling AND volume > 1.8x 20-bar avg
# - Exit when Alligator lines re-converge (Teeth crosses Jaw) indicating trend exhaustion
# - Uses 1d EMA50 for trend filter to avoid counter-trend trades
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-25 trades/year on 12h timeframe (50-100 total over 4 years)
# - Williams Alligator catches strong trends while filtering choppy markets

name = "12h_1d_williams_alligator_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute Williams Alligator from 12h data
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    
    # Williams Alligator: Smoothed Moving Average (SMMA)
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price = (high_12h + low_12h) / 2
    
    # Calculate SMMA (Smoothed Moving Average)
    def smma(data, period):
        result = np.full_like(data, np.nan, dtype=np.float64)
        if len(data) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Shift the lines as per Alligator definition
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute volume confirmation: > 1.8x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.8 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_20_avg[i])):
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
            bullish = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
            # Bearish alignment: Lips < Teeth < Jaw
            bearish = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
            
            # Long when bullish alignment AND 1d uptrend with volume spike
            if bullish and (close_12h[i] > ema50_1d_aligned[i]) and vol_spike.iloc[i]:
                position = 1
                signals[i] = 0.25
            # Short when bearish alignment AND 1d downtrend with volume spike
            elif bearish and (close_12h[i] < ema50_1d_aligned[i]) and vol_spike.iloc[i]:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit when Alligator re-converges
            # Exit when Teeth crosses Jaw (trend exhaustion)
            exit_signal = False
            if position == 1:  # Long position
                if teeth[i] <= jaw[i]:  # Teeth crossed below Jaw
                    exit_signal = True
            elif position == -1:  # Short position
                if teeth[i] >= jaw[i]:  # Teeth crossed above Jaw
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