#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Donchian Breakout + Weekly Volume Confirmation + ATR Filter
# Hypothesis: Donchian breakouts capture strong trends; weekly volume confirms institutional participation.
# ATR filter avoids whipsaws in low volatility. Works in bull via upside breakouts + volume + uptrend,
# in bear via downside breakouts + volume + downtrend. Target: 30-100 total trades over 4 years (7-25/year).

name = "1d_donchian_breakout_1w_volume_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Donchian Channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly volume confirmation: volume > 1.5x 20-period average
    vol_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(vol_1w).rolling(window=20, min_periods=10).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    vol_spike = volume > (1.5 * vol_ma_1w_aligned)
    
    # ATR (14-period) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(vol_ma_1w_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid low volatility whipsaws
        vol_filter = atr[i] > 0.01 * close[i]  # ATR > 1% of price
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR volatility drops
            if close[i] < low_20[i] or not vol_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR volatility drops
            if close[i] > high_20[i] or not vol_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_filter:
                # Long: price breaks above Donchian high with weekly volume spike
                if close[i] > high_20[i] and vol_spike[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian low with weekly volume spike
                elif close[i] < low_20[i] and vol_spike[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals