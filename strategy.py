#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w trend filter and volume confirmation
# - Williams Alligator: Jaw (EMA13,8), Teeth (EMA8,5), Lips (EMA5,3)
# - Long when Lips > Teeth > Jaw (bullish alignment) AND 1w close > 1w EMA50 AND volume > 1.5x 20-bar avg
# - Short when Lips < Teeth < Jaw (bearish alignment) AND 1w close < 1w EMA50 AND volume > 1.5x 20-bar avg
# - Exit when Alligator lines re-cross (Lips crosses Teeth) indicating trend weakness
# - Uses 1w EMA50 for stronger trend filter to avoid whipsaws
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-25 trades/year on 12h timeframe (50-100 total over 4 years)
# - Williams Alligator excels in trending markets; 1w filter ensures we only trade with primary trend

name = "12h_1w_williams_alligator_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Pre-compute Williams Alligator on 12h data
    # Williams Alligator: Jaw (13-period, 8-shift), Teeth (8-period, 5-shift), Lips (5-period, 3-shift)
    close = prices['close'].values
    
    # Calculate smoothed moving averages (SMMA) - using EMA as approximation
    jaw_raw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth_raw = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips_raw = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Apply shifts (Jaw: 8 bars, Teeth: 5 bars, Lips: 3 bars)
    jaw = np.full_like(jaw_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    
    jaw[8:] = jaw_raw[:-8]
    teeth[5:] = teeth_raw[:-5]
    lips[3:] = lips_raw[:-3]
    
    # Align Alligator lines to LTF (they're already on 12h, so just handle NaN)
    jaw_aligned = jaw
    teeth_aligned = teeth
    lips_aligned = lips
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Bullish Alligator: Lips > Teeth > Jaw
            bullish_align = (lips_aligned[i] > teeth_aligned[i] and 
                           teeth_aligned[i] > jaw_aligned[i])
            # Bearish Alligator: Lips < Teeth < Jaw
            bearish_align = (lips_aligned[i] < teeth_aligned[i] and 
                           teeth_aligned[i] < jaw_aligned[i])
            
            # Long when bullish alignment AND 1w uptrend with volume spike
            if (bullish_align and 
                prices['close'].iloc[i] > ema50_1w_aligned[i] and
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when bearish alignment AND 1w downtrend with volume spike
            elif (bearish_align and 
                  prices['close'].iloc[i] < ema50_1w_aligned[i] and
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit on Alligator re-cross
            # Exit when Lips crosses Teeth (trend weakness signal)
            exit_signal = False
            if position == 1:  # Long position
                # Exit bullish: Lips crosses below Teeth
                if lips_aligned[i] < teeth_aligned[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                # Exit bearish: Lips crosses above Teeth
                if lips_aligned[i] > teeth_aligned[i]:
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