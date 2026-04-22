#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian channel breakout with 1d VWAP trend filter and volume confirmation
    # Donchian(20) captures breakouts in trending markets
    # 1d VWAP filters for institutional trend direction (works in bull/bear)
    # Volume spike (2x 20-period MA) confirms breakout strength
    # Exit on opposite Donchian band touch for symmetry
    # Target: 20-50 trades/year to avoid fee drag
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for VWAP trend filter
    df_1d = get_htf_data(prices, '1d')
    # Calculate VWAP: cumulative(typical_price * volume) / cumulative(volume)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap = (typical_price * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_values = vwap.values
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_values)
    
    # Donchian Channel (20-period) on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above upper Donchian + volume spike + close above VWAP (uptrend)
            if close[i] > donchian_high[i] and vol_spike[i] and close[i] > vwap_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Donchian + volume spike + close below VWAP (downtrend)
            elif close[i] < donchian_low[i] and vol_spike[i] and close[i] < vwap_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Touch opposite Donchian band (mean reversion)
            if position == 1:
                if close[i] < donchian_low[i]:  # Touch lower band
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > donchian_high[i]:  # Touch upper band
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_1dVWAP_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0