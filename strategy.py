#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d volume spike + chop regime filter
# - Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs of median price
# - Long when Lips > Teeth > Jaw (bullish alignment) AND 1d volume > 2x 20-period average AND chop < 61.8 (trending regime)
# - Short when Lips < Teeth < Jaw (bearish alignment) AND 1d volume > 2x 20-period average AND chop < 61.8 (trending regime)
# - Exit when Alligator lines crossover (Lips crosses Teeth) OR chop > 61.8 (range regime)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Alligator identifies trend direction and strength
# - Volume confirmation ensures breakouts have conviction
# - Chop filter avoids whipsaws in ranging markets

name = "12h_1d_alligator_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Median price for Alligator
    median_price = (high + low) / 2
    
    # Williams Alligator components (SMAs of median price)
    def sma(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.mean(arr[i - window + 1:i + 1])
        return result
    
    jaw = sma(median_price, 13)  # Jaw: 13-period, 8 bars ahead
    teeth = sma(median_price, 8)  # Teeth: 8-period, 5 bars ahead
    lips = sma(median_price, 5)   # Lips: 5-period, 3 bars ahead
    
    # Shift jaw and teeth to align with lips (Alligator alignment)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    # Lips remains unshifted (reference)
    
    # Pre-compute 12h chop regime (EWMA of True Range)
    def true_range(h, l, c):
        tr1 = h - l
        tr2 = np.abs(h - np.roll(c, 1))
        tr3 = np.abs(l - np.roll(c, 1))
        tr1[0] = 0
        tr2[0] = 0
        tr3[0] = 0
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    tr = true_range(high, low, close)
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Chop index: ATR normalized by price range over period
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    price_range = max_high - min_low
    chop = np.where(price_range > 0, (atr * 100) / price_range, 100)
    
    # Trending regime: chop < 61.8
    trending_regime = chop < 61.8
    
    # Pre-compute 1d volume confirmation
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = df_1d['volume'].values > (2.0 * vol_ma_1d)
    
    # Align HTF indicators to 12h timeframe
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    trending_regime_aligned = align_htf_to_ltf(prices, df_1d, trending_regime)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips[i]) or
            np.isnan(volume_spike_1d_aligned[i]) or np.isnan(trending_regime_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Bullish Alligator alignment: Lips > Teeth > Jaw
            bullish = lips[i] > teeth_shifted[i] > jaw_shifted[i]
            # Bearish Alligator alignment: Lips < Teeth < Jaw
            bearish = lips[i] < teeth_shifted[i] < jaw_shifted[i]
            
            if bullish and volume_spike_1d_aligned[i] and trending_regime_aligned[i]:
                position = 1
                signals[i] = 0.25
            elif bearish and volume_spike_1d_aligned[i] and trending_regime_aligned[i]:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Alligator crossover OR chop > 61.8 (range regime)
            lips_cross_teeth = (position == 1 and lips[i] < teeth_shifted[i]) or \
                               (position == -1 and lips[i] > teeth_shifted[i])
            exit_chop = not trending_regime_aligned[i]  # chop > 61.8
            
            if lips_cross_teeth or exit_chop:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals