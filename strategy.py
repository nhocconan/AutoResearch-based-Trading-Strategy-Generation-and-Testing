#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# - Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs
# - Long when Lips > Teeth > Jaw (bullish alignment) AND 1d close > 1d EMA50 AND volume > 1.5x 20-bar avg
# - Short when Lips < Teeth < Jaw (bearish alignment) AND 1d close < 1d EMA50 AND volume > 1.5x 20-bar avg
# - Exit when Alligator lines cross (Lips-Teeth or Teeth-Jaw crossover)
# - Uses 1d EMA50 for stronger trend filter to avoid whipsaws
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-25 trades/year on 12h timeframe (50-100 total over 4 years)
# - Williams Alligator catches strong trends while avoiding choppy markets

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
    # Jaw: 13-period SMA, shifted 8 bars
    jaw = pd.Series(close_12h).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMA, shifted 5 bars
    teeth = pd.Series(close_12h).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMA, shifted 3 bars
    lips = pd.Series(close_12h).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines (already calculated on 12h data, no additional alignment needed)
    # But we need to ensure proper indexing - the shift operations are already done
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_20_avg[i])):
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
            bullish = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
            # Bearish alignment: Lips < Teeth < Jaw
            bearish = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
            
            # Long when bullish alignment AND 1d uptrend with volume spike
            if bullish and (prices['close'].iloc[i] > ema50_1d_aligned[i]) and vol_spike.iloc[i]:
                position = 1
                signals[i] = 0.25
            # Short when bearish alignment AND 1d downtrend with volume spike
            elif bearish and (prices['close'].iloc[i] < ema50_1d_aligned[i]) and vol_spike.iloc[i]:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit on Alligator crossover
            # Exit when Lips-Teeth crossover OR Teeth-Jaw crossover
            exit_signal = False
            if position == 1:  # Long position
                # Exit if bullish alignment breaks
                if not (lips[i] > teeth[i] and teeth[i] > jaw[i]):
                    exit_signal = True
            elif position == -1:  # Short position
                # Exit if bearish alignment breaks
                if not (lips[i] < teeth[i] and teeth[i] < jaw[i]):
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