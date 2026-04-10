#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# - Alligator: Jaw (EMA13, 8 periods), Teeth (EMA8, 5 periods), Lips (EMA5, 3 periods)
# - Long when Lips > Teeth > Jaw (bullish alignment) AND 1d close > EMA50 AND volume > 1.5x avg
# - Short when Lips < Teeth < Jaw (bearish alignment) AND 1d close < EMA50 AND volume > 1.5x avg
# - Exit when alignment breaks (Lips crosses Teeth or Teeth crosses Jaw)
# - Uses discrete position sizing (0.25) to control drawdown
# - Targets ~12-25 trades/year (50-100 total over 4 years) to avoid fee drag
# - Alligator identifies trend absence (sleeping), formation, and strength
# - Works in both bull (strong uptrend alignment) and bear (strong downtrend alignment)
# - Volume confirmation prevents false signals
# - 1d EMA50 filter ensures alignment with higher timeframe trend

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
    
    # Pre-compute Williams Alligator components (using 12h data)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Alligator: Jaw (EMA13, 8 periods), Teeth (EMA8, 5 periods), Lips (EMA5, 3 periods)
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values  # Blue line
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values   # Red line
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values    # Green line
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(13, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_20_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: Lips > Teeth > Jaw (bullish alignment) AND 1d uptrend AND volume spike
            if (lips[i] > teeth[i] and 
                teeth[i] > jaw[i] and 
                close[i] > ema50_1d_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: Lips < Teeth < Jaw (bearish alignment) AND 1d downtrend AND volume spike
            elif (lips[i] < teeth[i] and 
                  teeth[i] < jaw[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions: when Alligator alignment breaks
            # Long exits when Lips <= Teeth or Teeth <= Jaw
            # Short exits when Lips >= Teeth or Teeth >= Jaw
            if position == 1:
                if lips[i] <= teeth[i] or teeth[i] <= jaw[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:
                if lips[i] >= teeth[i] or teeth[i] >= jaw[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals