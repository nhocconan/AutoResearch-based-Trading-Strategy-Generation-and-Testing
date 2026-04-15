#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with 1d volume spike and chop regime filter.
# Camarilla pivot levels provide institutional support/resistance. Breakouts with volume
# indicate institutional participation. Chop filter avoids whipsaws in ranging markets.
# Works in bull (breakouts continue) and bear (fades at S1/R1) regimes.
# Target: 12-37 trades/year (50-150 over 4 years) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (R1, S1, R4, S4)
    # Based on previous day's OHLC
    ph = df_1d['high'].shift(1).values
    pl = df_1d['low'].shift(1).values
    pc = df_1d['close'].shift(1).values
    
    # Camarilla calculations
    camarilla_range = ph - pl
    r1 = pc + (camarilla_range * 1.1 / 12)
    s1 = pc - (camarilla_range * 1.1 / 12)
    r4 = pc + (camarilla_range * 1.1 / 2)
    s4 = pc - (camarilla_range * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe (completed 1d bar only)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 1d volume spike (volume > 1.5x 20-period EMA)
    vol_ema = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = df_1d['volume'].values > (1.5 * vol_ema)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    # Calculate 1d choppiness index (CHOP) for regime filter
    # CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    # We want trending markets (CHOP < 50) for breakout strategies
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    high_low_range = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values - \
                     pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / high_low_range) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_spike_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when market is trending (CHOP < 50)
        regime_filter = chop_aligned[i] < 50
        
        # Volume confirmation: current 1d bar had volume spike
        vol_confirm = vol_spike_aligned[i] > 0.5
        
        # Long breakout: price breaks above R1 with volume and in trending regime
        if (close[i] > r1_aligned[i] and vol_confirm and regime_filter):
            signals[i] = 0.25
            
        # Short breakdown: price breaks below S1 with volume and in trending regime
        elif (close[i] < s1_aligned[i] and vol_confirm and regime_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_Volume_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0