#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1d trend filter and volume confirmation
# - Long when price breaks above latest bearish fractal high AND 1d close > 1d EMA50 AND volume > 1.8x 20-bar avg
# - Short when price breaks below latest bullish fractal low AND 1d close < 1d EMA50 AND volume > 1.8x 20-bar avg
# - Exit when price crosses 1d EMA50 (trend reversal signal)
# - Uses 1d EMA50 for trend filter to avoid counter-trend trades
# - Williams Fractals provide natural support/resistance levels from price action
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-30 trades/year on 6h timeframe (50-120 total over 4 years)

name = "6h_1d_williams_fractal_breakout_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute Williams Fractals from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams Fractals: 5-bar pattern
    # Bearish fractal: high[n-2] is highest of [n-4, n-3, n-2, n-1, n]
    # Bullish fractal: low[n-2] is lowest of [n-4, n-3, n-2, n-1, n]
    bearish_fractal = np.full(len(high_1d), np.nan)
    bullish_fractal = np.full(len(low_1d), np.nan)
    
    for i in range(2, len(high_1d) - 2):
        # Bearish fractal: middle bar has highest high
        if (high_1d[i] >= high_1d[i-2] and high_1d[i] >= high_1d[i-1] and 
            high_1d[i] >= high_1d[i+1] and high_1d[i] >= high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        # Bullish fractal: middle bar has lowest low
        if (low_1d[i] <= low_1d[i-2] and low_1d[i] <= low_1d[i-1] and 
            low_1d[i] <= low_1d[i+1] and low_1d[i] <= low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to LTF with additional delay for fractals (need 2 extra bars for confirmation)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute volume confirmation: > 1.8x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.8 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above bearish fractal high AND 1d uptrend with volume spike
            if (prices['close'].iloc[i] > bearish_fractal_aligned[i] and 
                prices['close'].iloc[i] > ema50_1d_aligned[i] and  # price above 1d EMA50
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below bullish fractal low AND 1d downtrend with volume spike
            elif (prices['close'].iloc[i] < bullish_fractal_aligned[i] and 
                  prices['close'].iloc[i] < ema50_1d_aligned[i] and  # price below 1d EMA50
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit on trend reversal
            # Exit when price crosses 1d EMA50 (trend reversal)
            exit_signal = False
            if position == 1:  # Long position
                if prices['close'].iloc[i] <= ema50_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                if prices['close'].iloc[i] >= ema50_1d_aligned[i]:
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