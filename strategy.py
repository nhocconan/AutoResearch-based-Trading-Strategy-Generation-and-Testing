#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + volume confirmation + 1d ATR regime filter
# - Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs of median price
# - Long when Lips > Teeth > Jaw (bullish alignment) AND volume > 1.5x 24-period average AND 1d ATR(14) < 20-period median ATR (low volatility)
# - Short when Lips < Teeth < Jaw (bearish alignment) AND volume > 1.5x 24-period average AND 1d ATR(14) < 20-period median ATR
# - Exit when Alligator lines cross (Lips crosses Teeth) or ATR regime shifts to high volatility
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Alligator identifies trend initiation and continuation with built-in smoothing
# - Volume confirmation reduces false signals during low participation periods
# - ATR filter ensures we trade during low volatility periods when trends are more sustainable

name = "12h_1d_alligator_atr_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 12h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Median price for Alligator calculation
    median_price = (high + low) / 2
    
    # Williams Alligator components (SMAs of median price)
    # Jaw: 13-period SMA, 8 periods forward
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # shift forward 8 periods
    
    # Teeth: 8-period SMA, 5 periods forward
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # shift forward 5 periods
    
    # Lips: 5-period SMA, 3 periods forward
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # shift forward 3 periods
    
    # Pre-compute 12h volume confirmation (24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1d ATR(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) using Wilder's smoothing
    atr_1d = np.zeros_like(tr)
    atr_1d[13] = np.mean(tr[1:14])  # First ATR value
    for i in range(14, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # ATR regime: low volatility when current ATR < median of last 20 ATR values
    atr_median_20 = pd.Series(atr_1d).rolling(window=20, min_periods=20).median().values
    low_vol_regime = atr_1d < atr_median_20
    
    # Align HTF indicators to 12h timeframe
    low_vol_regime_aligned = align_htf_to_ltf(prices, df_1d, low_vol_regime)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(low_vol_regime_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Alligator alignment conditions
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: bullish Alligator alignment AND low volatility regime AND volume spike
            if (bullish_alignment and 
                low_vol_regime_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: bearish Alligator alignment AND low volatility regime AND volume spike
            elif (bearish_alignment and 
                  low_vol_regime_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Alligator cross OR regime shift to high volatility
            lips_teeth_cross = (position == 1 and lips[i] < teeth[i]) or (position == -1 and lips[i] > teeth[i])
            high_vol_regime = not low_vol_regime_aligned[i]
            
            if lips_teeth_cross or high_vol_regime:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals