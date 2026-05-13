#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and choppiness regime filter.
# Long when price breaks above Camarilla R3 AND 1d volume > 1.5x 20-period average AND CHOP(14) > 61.8 (ranging market).
# Short when price breaks below Camarilla S3 AND same volume/chop conditions.
# Uses discrete position sizing (0.25) to limit fee churn. Designed for 12-37 trades/year by requiring confluence of
# price level breakout, volume confirmation, and ranging regime - proven to work in both bull (mean reversion) and bear (range) markets.

name = "12h_Camarilla_R3S3_Breakout_1dVol_Chop_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels (R3, S3) from prior 1d bar
    # R3 = close + 1.1*(high - low)/2
    # S3 = close - 1.1*(high - low)/2
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 12h timeframe (available after 1d bar close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d volume spike: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.5 * vol_ma20)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    # Choppiness Index on 12h timeframe: CHOP > 61.8 = ranging (mean reversion regime)
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop[0:13] = np.nan  # First 13 bars invalid
    chop_regime = chop > 61.8  # True when ranging
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data
        if np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or \
           np.isnan(vol_spike_aligned[i]) or np.isnan(chop_regime[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 AND volume spike AND ranging regime
            if close[i] > camarilla_r3_aligned[i] and vol_spike_aligned[i] > 0.5 and chop_regime[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 AND volume spike AND ranging regime
            elif close[i] < camarilla_s3_aligned[i] and vol_spike_aligned[i] > 0.5 and chop_regime[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 (contrary signal) OR chop regime ends (trend begins)
            if close[i] < camarilla_s3_aligned[i] or chop_regime[i] == False:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 OR chop regime ends
            if close[i] > camarilla_r3_aligned[i] or chop_regime[i] == False:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals