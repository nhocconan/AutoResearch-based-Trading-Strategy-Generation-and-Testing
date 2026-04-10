#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + ADX trend strength + volume confirmation
# - Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs of median price
# - Trend up when Lips > Teeth > Jaw, down when Lips < Teeth < Jaw
# - ADX > 25 confirms strong trend (avoid choppy markets)
# - Volume > 1.2x 20-period average confirms participation
# - Enter long when Alligator bullish + ADX>25 + volume confirmation
# - Enter short when Alligator bearish + ADX>25 + volume confirmation
# - Exit when Alligator reverses (Teeth crosses Lips) OR ADX < 20 (trend weakening)
# - Targets 12-25 trades/year (50-100 total over 4 years) to avoid fee drag
# - Williams Alligator identifies trend inception and continuation
# - ADX filters out ranging markets where Alligator whipsaws
# - Volume confirmation ensures breakouts have conviction

name = "6h_1d_williams_alligator_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute Williams Alligator on 6h data
    median_price = (prices['high'] + prices['low']) / 2.0
    jaw = median_price.rolling(window=13, min_periods=13).mean().rolling(window=8, min_periods=8).mean().values
    teeth = median_price.rolling(window=8, min_periods=8).mean().rolling(window=5, min_periods=5).mean().values
    lips = median_price.rolling(window=5, min_periods=5).mean().rolling(window=3, min_periods=3).mean().values
    
    # Alligator relationships
    alligator_bullish = (lips > teeth) & (teeth > jaw)
    alligator_bearish = (lips < teeth) & (teeth < jaw)
    
    # Pre-compute ADX on 1d for trend strength (HTF)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index 0
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+ , DM- (Wilder's smoothing = EMA with alpha=1/period)
    period = 14
    alpha = 1.0 / period
    tr_period = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean().values
    dm_plus_period = pd.Series(dm_plus).ewm(alpha=alpha, adjust=False).mean().values
    dm_minus_period = pd.Series(dm_minus).ewm(alpha=alpha, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_period / tr_period
    di_minus = 100 * dm_minus_period / tr_period
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=alpha, adjust=False).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute volume confirmation: > 1.2x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.2 * volume_20_avg)
    
    # Exit conditions
    alligator_weak = ~(alligator_bullish | alligator_bearish)  # Transition state
    adx_weak = adx_aligned < 20  # Trend weakening
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after Alligator warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_20_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Alligator bullish + strong trend + volume confirmation
            if (alligator_bullish[i] and 
                adx_aligned[i] > 25 and 
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Alligator bearish + strong trend + volume confirmation
            elif (alligator_bearish[i] and 
                  adx_aligned[i] > 25 and 
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Alligator reverses (Teeth crosses Lips)
            # 2. ADX drops below 20 (trend weakening)
            if position == 1:  # Long position
                if (alligator_weak[i] or 
                    adx_weak[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (alligator_weak[i] or 
                    adx_weak[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals