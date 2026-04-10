#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# - Williams Alligator: Jaw (EMA13, 8-period shift), Teeth (EMA8, 5-period shift), Lips (EMA5, 3-period shift)
# - Long when Lips > Teeth > Jaw (bullish alignment) with volume > 1.3x average AND daily close > daily EMA50
# - Short when Lips < Teeth < Jaw (bearish alignment) with volume > 1.3x average AND daily close < daily EMA50
# - Exit when Alligator lines converge (Lips crosses Teeth or Jaw) or volume drops below average
# - Daily trend filter ensures alignment with major trend
# - Volume confirmation prevents false signals
# - Targets 12-25 trades/year (50-100 total over 4 years) to avoid fee drag
# - Williams Alligator identifies trending vs ranging markets; works in both bull and bear when combined with volume/trend filters

name = "12h_1d_alligator_volume_trend_v1"
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
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute Williams Alligator on 12h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Jaw: EMA(13) shifted by 8 bars
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # shift right by 8 (look back)
    jaw[:8] = np.nan  # first 8 values invalid
    
    # Teeth: EMA(8) shifted by 5 bars
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # shift right by 5 (look back)
    teeth[:5] = np.nan  # first 5 values invalid
    
    # Lips: EMA(5) shifted by 3 bars
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # shift right by 3 (look back)
    lips[:3] = np.nan  # first 3 values invalid
    
    # Pre-compute volume confirmation: > 1.3x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.3 * volume_20_avg)
    
    # Pre-compute volume filter: < average volume for exit
    vol_normal = prices['volume'] < volume_20_avg
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(13, n):  # start after jaw warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_20_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for new entries
            # Bullish alignment: Lips > Teeth > Jaw
            bullish = lips[i] > teeth[i] > jaw[i]
            # Bearish alignment: Lips < Teeth < Jaw
            bearish = lips[i] < teeth[i] < jaw[i]
            
            # Long entry: bullish alignment + volume spike + daily uptrend
            if bullish and vol_spike.iloc[i] and prices['close'].iloc[i] > ema50_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: bearish alignment + volume spike + daily downtrend
            elif bearish and vol_spike.iloc[i] and prices['close'].iloc[i] < ema50_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Alligator lines converge (Lips crosses Teeth or Jaw)
            # 2. Volume drops below average (loss of momentum)
            lips_cross_teeth = (lips[i-1] - teeth[i-1]) * (lips[i] - teeth[i]) <= 0
            lips_cross_jaw = (lips[i-1] - jaw[i-1]) * (lips[i] - jaw[i]) <= 0
            convergence = lips_cross_teeth or lips_cross_jaw
            
            if position == 1:  # Long position
                if convergence or vol_normal.iloc[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if convergence or vol_normal.iloc[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals