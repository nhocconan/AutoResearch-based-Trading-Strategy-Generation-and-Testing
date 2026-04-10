#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal breakout with 1d trend filter and volume confirmation
# - Long when price breaks above the most recent bullish Williams fractal high with volume > 1.8x average AND 1d close > 1d EMA50
# - Short when price breaks below the most recent bearish Williams fractal low with volume > 1.8x average AND 1d close < 1d EMA50
# - Exit when price crosses the 1d EMA50 (trend reversal) OR volume drops below 0.7x average
# - Uses 1d EMA50 trend filter to avoid counter-trend trades in bear markets (2025+)
# - Williams Fractals provide structure based on actual price swings, reducing false breakouts
# - Volume confirmation (1.8x) and trend filter target 12-25 trades/year (48-100 total over 4 years)
# - Designed to work in both bull and bear regimes by aligning with 1d trend

name = "12h_1d_williams_fractal_breakout_volume_trend_v1"
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
    
    # Pre-compute Williams Fractals on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams Fractals: bearish (high) and bullish (low) patterns
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-1] > high[n-3] and high[n-1] > high[n+1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-1] < low[n-3] and low[n-1] < low[n+1]
    bearish_fractal = np.full(len(high_1d), np.nan)
    bullish_fractal = np.full(len(low_1d), np.nan)
    
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i] < high_1d[i-1] and
            high_1d[i-3] < high_1d[i-1] and
            high_1d[i+1] < high_1d[i-1]):
            bearish_fractal[i-1] = high_1d[i-1]  # Store the fractal high at the center bar
            
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i] > low_1d[i-1] and
            low_1d[i-3] > low_1d[i-1] and
            low_1d[i+1] > low_1d[i-1]):
            bullish_fractal[i-1] = low_1d[i-1]  # Store the fractal low at the center bar
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Pre-compute volume confirmation: > 1.8x 30-period average
    volume_30_avg = prices['volume'].rolling(window=30, min_periods=30).mean().values
    vol_spike = prices['volume'] > (1.8 * volume_30_avg)
    
    # Pre-compute volume filter: < 0.7x average volume for exit (loss of momentum)
    vol_weak = prices['volume'] < (0.7 * volume_30_avg)
    
    # Align HTF data to 12h timeframe
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    last_bullish_fractal = np.nan  # Most recent completed bullish fractal low
    last_bearish_fractal = np.nan  # Most recent completed bearish fractal high
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_30_avg[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Update most recent completed fractals (using aligned data with proper delay)
        if not np.isnan(bullish_fractal_aligned[i]):
            last_bullish_fractal = bullish_fractal_aligned[i]
        if not np.isnan(bearish_fractal_aligned[i]):
            last_bearish_fractal = bearish_fractal_aligned[i]
        
        if position == 0:  # Flat - look for new breakout entries
            # Long breakout: price > most recent bullish fractal low with volume spike AND 1d uptrend
            if (not np.isnan(last_bullish_fractal) and 
                prices['close'].iloc[i] > last_bullish_fractal and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown: price < most recent bearish fractal high with volume spike AND 1d downtrend
            elif (not np.isnan(last_bearish_fractal) and 
                  prices['close'].iloc[i] < last_bearish_fractal and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0  # Stay flat
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price crosses 1d EMA50 (trend reversal)
            # 2. Volume drops below 0.7x average (loss of momentum)
            if position == 1:  # Long position
                if (prices['close'].iloc[i] < ema50_1d_aligned[i] or 
                    vol_weak.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (prices['close'].iloc[i] > ema50_1d_aligned[i] or 
                    vol_weak.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals