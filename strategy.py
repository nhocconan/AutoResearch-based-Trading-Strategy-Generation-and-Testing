#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ADX trend filter and volume confirmation.
# In bull markets, captures breakouts; in bear markets, ADX>25 filters for strong trends only.
# Volume confirmation reduces false breakouts. Target: 15-25 trades/year.
def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d Donchian channels (20-period) - calculated once
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # 1d ADX for trend strength (14-period)
    # Calculate TR, +DM, -DM
    tr1 = pd.Series(high_1d).rolling(2).max() - pd.Series(low_1d).rolling(2).min()
    tr2 = abs(pd.Series(high_1d).rolling(2).max() - pd.Series(close).shift(1))
    tr3 = abs(pd.Series(low_1d).rolling(2).min() - pd.Series(close).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    plus_dm = pd.Series(high_1d).diff()
    minus_dm = pd.Series(low_1d).diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/14, adjust=False).mean()
    
    adx_values = adx.values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Volume confirmation: current > 1.5x median of last 20 periods
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: close breaks above Donchian high + ADX > 25 + volume confirmation
        if (close[i] > donchian_high_aligned[i] and 
            adx_aligned[i] > 25 and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: close breaks below Donchian low + ADX > 25 + volume confirmation
        elif (close[i] < donchian_low_aligned[i] and 
              adx_aligned[i] > 25 and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: close crosses back inside Donchian channels (mean reversion)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < donchian_high_aligned[i]) or
               (signals[i-1] == -0.25 and close[i] > donchian_low_aligned[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_Donchian_ADX_Volume"
timeframe = "12h"
leverage = 1.0