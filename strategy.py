#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d trend filter and volume confirmation
# - Williams Alligator consists of three SMAs (jaw, teeth, lips) representing balance lines
# - Long when: lips > teeth > jaw (bullish alignment) AND price > 1d EMA50 (uptrend) AND volume > 1.3x average
# - Short when: lips < teeth < jaw (bearish alignment) AND price < 1d EMA50 (downtrend) AND volume > 1.3x average
# - Exit when: Alligator lines converge (|lips - jaw| < 0.5 * ATR) OR volume drops below average
# - Williams Alligator identifies trend absence (convergence) and presence (divergence)
# - 1d EMA50 filter ensures alignment with higher timeframe trend
# - Volume confirmation prevents false signals during low participation
# - Targets 12-25 trades/year (48-100 total over 4 years) to avoid fee drag
# - Works in both bull and bear markets by capturing genuine trend phases with filters

name = "6h_1d_alligator_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator: 3 SMAs with different periods and offsets
    # Jaw: SMA(13, 8) - blue line
    # Teeth: SMA(8, 5) - red line  
    # Lips: SMA(5, 3) - green line
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Jaw (13-period SMA, offset 8 bars forward)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth (8-period SMA, offset 5 bars forward)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips (5-period SMA, offset 3 bars forward)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # 1d EMA(50) for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # ATR(14) for exit condition (convergence measurement)
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: > 1.3x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.3 * volume_20_avg)
    
    # Volume filter: < average volume for exit
    vol_normal = prices['volume'] < volume_20_avg
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(atr[i]) or
            np.isnan(volume_20_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for new entries
            # Bullish Alligator alignment: lips > teeth > jaw
            bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
            # Bearish Alligator alignment: lips < teeth < jaw
            bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
            
            # Long entry: bullish alignment + price > 1d EMA50 + volume spike
            if (bullish_alignment and 
                prices['close'].iloc[i] > ema50_1d_aligned[i] and
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: bearish alignment + price < 1d EMA50 + volume spike
            elif (bearish_alignment and
                  prices['close'].iloc[i] < ema50_1d_aligned[i] and
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Alligator convergence: |lips - jaw| < 0.5 * ATR (trend weakening)
            # 2. Volume drops below average (loss of momentum)
            alligator_convergence = np.abs(lips[i] - jaw[i]) < (0.5 * atr[i])
            
            if position == 1:  # Long position
                if (alligator_convergence or vol_normal.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (alligator_convergence or vol_normal.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals