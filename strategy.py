#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w trend filter and volume confirmation
# - Long when price > Alligator Jaw AND Jaw > Teeth > Lips (bullish alignment) AND volume > 1.5x 20-bar avg
# - Short when price < Alligator Jaw AND Jaw < Teeth < Lips (bearish alignment) AND volume > 1.5x 20-bar avg
# - Exit when Alligator lines cross (Jaw-Teeth or Teeth-Lips crossover) indicating trend weakness
# - Uses 1w EMA50 for trend filter to avoid counter-trend trades in bear markets
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 15-25 trades/year on 1d timeframe (60-100 total over 4 years)
# - Williams Alligator excels in trending markets; 1w filter improves bear market performance

name = "1d_1w_alligator_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Pre-compute Williams Alligator from daily data (Jaw=13, Teeth=8, Lips=5)
    # Alligator uses SMAs with future shift (but we align properly to avoid look-ahead)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Jaw: 13-period SMMA shifted 8 bars forward
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMMA shifted 5 bars forward  
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMMA shifted 3 bars forward
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup for Alligator shifts
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Bullish alignment: Jaw > Teeth > Lips AND price above Jaw
            bullish = (jaw[i] > teeth[i] > lips[i]) and (prices['close'].iloc[i] > jaw[i])
            # Bearish alignment: Jaw < Teeth < Lips AND price below Jaw
            bearish = (jaw[i] < teeth[i] < lips[i]) and (prices['close'].iloc[i] < jaw[i])
            
            # Long when bullish alignment with 1w uptrend and volume spike
            if bullish and (prices['close'].iloc[i] > ema50_1w_aligned[i]) and vol_spike.iloc[i]:
                position = 1
                signals[i] = 0.25
            # Short when bearish alignment with 1w downtrend and volume spike
            elif bearish and (prices['close'].iloc[i] < ema50_1w_aligned[i]) and vol_spike.iloc[i]:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit on Alligator cross (trend weakness)
            # Exit when Jaw-Teeth cross OR Teeth-Lips cross (indicates trend losing momentum)
            exit_signal = False
            if position == 1:  # Long position
                if (jaw[i] <= teeth[i]) or (teeth[i] <= lips[i]):
                    exit_signal = True
            elif position == -1:  # Short position
                if (jaw[i] >= teeth[i]) or (teeth[i] >= lips[i]):
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