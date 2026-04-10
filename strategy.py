#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w trend filter and volume confirmation
# - Williams Alligator: Jaw (SMA13, 8-bar shift), Teeth (SMA8, 5-bar shift), Lips (SMA5, 3-bar shift)
# - Long when Lips > Teeth > Jaw (bullish alignment) AND price > Lips AND weekly close > weekly EMA20 AND volume > 1.5x average
# - Short when Lips < Teeth < Jaw (bearish alignment) AND price < Lips AND weekly close < weekly EMA20 AND volume > 1.5x average
# - Exit when Alligator lines cross (Lips crosses Teeth) OR volume drops below average
# - Weekly trend filter ensures alignment with major trend
# - Volume confirmation prevents false signals
# - Targets 20-30 trades/year (80-120 total over 4 years) to avoid fee drag
# - Williams Alligator is effective in trending markets and avoids whipsaws in ranging markets

name = "1d_1w_alligator_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute Williams Alligator on daily data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Jaw: SMA(13) shifted 8 bars
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: SMA(8) shifted 5 bars
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: SMA(5) shifted 3 bars
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Pre-compute 1w EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute volume filter: < average volume for exit
    vol_normal = prices['volume'] < volume_20_avg
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(13, n):  # Start after Alligator warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(volume_20_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for new entries
            # Bullish Alligator alignment: Lips > Teeth > Jaw
            bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
            # Bearish Alligator alignment: Lips < Teeth < Jaw
            bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
            
            # Long entry: bullish alignment AND price > Lips AND weekly uptrend AND volume spike
            if (bullish_alignment and 
                prices['close'].iloc[i] > lips[i] and 
                prices['close'].iloc[i] > ema20_1w_aligned[i] and
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: bearish alignment AND price < Lips AND weekly downtrend AND volume spike
            elif (bearish_alignment and 
                  prices['close'].iloc[i] < lips[i] and 
                  prices['close'].iloc[i] < ema20_1w_aligned[i] and
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Alligator lines cross (Lips crosses Teeth) - trend weakening
            # 2. Volume drops below average (loss of momentum)
            lips_teeth_cross = (lips[i] > teeth[i] and lips[i-1] <= teeth[i-1]) or \
                               (lips[i] < teeth[i] and lips[i-1] >= teeth[i-1])
            
            if position == 1:  # Long position
                if lips_teeth_cross or vol_normal.iloc[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if lips_teeth_cross or vol_normal.iloc[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals