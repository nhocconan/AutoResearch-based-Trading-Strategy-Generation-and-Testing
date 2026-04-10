#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation
# - Uses Alligator (Jaw=13, Teeth=8, Lips=5) to identify trendless markets
# - Long when price > Alligator Lips AND Lips > Teeth > Jaw (bullish alignment) AND volume > 1.5x average AND 1d close > 1d EMA50
# - Short when price < Alligator Lips AND Lips < Teeth < Jaw (bearish alignment) AND volume > 1.5x average AND 1d close < 1d EMA50
# - Exit when Alligator lines cross (trend weakness) OR volume < 0.7x average
# - Williams Alligator is effective in both trending and ranging markets, reducing false signals
# - Target: 25-35 trades/year (100-140 total over 4 years) to avoid fee drag
# - Works in BTC/ETH by filtering counter-trend trades in bear markets (2025+) with 1d EMA50

name = "4h_1d_alligator_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute volume filter: < 0.7x average volume for exit (loss of momentum)
    vol_weak = prices['volume'] < (0.7 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute Williams Alligator components
    # Alligator: Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA)
    close = prices['close'].values
    
    # SMMA (Smoothed Moving Average) calculation
    def smma(source, period):
        """Calculate Smoothed Moving Average"""
        if len(source) < period:
            return np.full_like(source, np.nan)
        result = np.full_like(source, np.nan)
        # First value is SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA*(period-1) + CURRENT_PRICE) / period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw = smma(close, 13)   # Jaw (Blue) - 13-period SMMA
    teeth = smma(close, 8)  # Teeth (Red) - 8-period SMMA
    lips = smma(close, 5)   # Lips (Green) - 5-period SMMA
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_20_avg[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Bullish alignment: Lips > Teeth > Jaw
            bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
            # Bearish alignment: Lips < Teeth < Jaw
            bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
            
            # Long entry: price > Lips AND bullish alignment AND volume spike AND 1d uptrend
            if (prices['close'].iloc[i] > lips[i] and 
                bullish_alignment and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price < Lips AND bearish alignment AND volume spike AND 1d downtrend
            elif (prices['close'].iloc[i] < lips[i] and 
                  bearish_alignment and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0  # Stay flat
        else:  # Have position - look for exit
            # Check for Alligator lines crossing (trend weakness)
            lips_teeth_cross = (lips[i] > teeth[i]) != (lips[i-1] > teeth[i-1]) if i > 0 else False
            teeth_jaw_cross = (teeth[i] > jaw[i]) != (teeth[i-1] > jaw[i-1]) if i > 0 else False
            alligator_cross = lips_teeth_cross or teeth_jaw_cross
            
            if position == 1:  # Long position
                if alligator_cross or vol_weak.iloc[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if alligator_cross or vol_weak.iloc[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals