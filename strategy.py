#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume spike and chop regime filter
# - Long when price breaks above Camarilla H3 level AND 1d volume > 2.0x 20-period 1d volume SMA AND chop > 61.8 (ranging market)
# - Short when price breaks below Camarilla L3 level AND 1d volume > 2.0x 20-period 1d volume SMA AND chop > 61.8
# - Exit: Price reversion to Camarilla Pivot level or opposite breakout with volume confirmation
# - Position sizing: 0.25 discrete level
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Camarilla levels from 1d provide institutional support/resistance, volume spike confirms institutional interest,
#   chop filter ensures mean-reversion behavior in ranging markets (avoids trending whipsaw)

name = "12h_1d_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels (based on previous 1d bar)
    # Camarilla: H4 = close + 1.5*(high-low), H3 = close + 1.0*(high-low), 
    #            L3 = close - 1.0*(high-low), L4 = close - 1.5*(high-low)
    # Pivot = (high + low + close) / 3
    df_1d['pivot'] = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    df_1d['range'] = df_1d['high'] - df_1d['low']
    df_1d['H3'] = df_1d['close'] + 1.0 * df_1d['range']
    df_1d['L3'] = df_1d['close'] - 1.0 * df_1d['range']
    
    # Align Camarilla levels to 12h timeframe (wait for completed 1d bar)
    camarilla_H3 = align_htf_to_ltf(prices, df_1d, df_1d['H3'].values)
    camarilla_L3 = align_htf_to_ltf(prices, df_1d, df_1d['L3'].values)
    camarilla_pivot = align_htf_to_ltf(prices, df_1d, df_1d['pivot'].values)
    
    # Calculate 1d volume spike confirmation (wait for completed 1d bar)
    volume_sma_20_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = align_htf_to_ltf(prices, df_1d, df_1d['volume'].values > 2.0 * volume_sma_20_1d)
    
    # Calculate 1d Chopiness Index for regime filter (wait for completed 1d bar)
    # Chop = 100 * log10(sum(ATR(14)) / (n * log10(n))) / log10(n)
    # Simplified: Chop > 61.8 = ranging, Chop < 38.2 = trending
    tr1 = pd.Series(df_1d['high'] - df_1d['low'])
    tr2 = pd.Series(np.abs(df_1d['high'] - np.roll(df_1d['close'], 1)))
    tr3 = pd.Series(np.abs(df_1d['low'] - np.roll(df_1d['close'], 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    chop = 100 * np.log10(atr_14.rolling(window=14, min_periods=14).sum() / 
                           (14 * np.log10(14))) / np.log10(14)
    chop_align = align_htf_to_ltf(prices, df_1d, chop.values)
    chop_regime = chop_align > 61.8  # Ranging market (mean reversion favorable)
    
    # Track entry price for exit logic
    entry_price = np.full(n, np.nan)
    
    for i in range(60, n):  # Warmup period for HTF indicators
        # Skip if any required data is invalid
        if (np.isnan(camarilla_H3[i]) or np.isnan(camarilla_L3[i]) or 
            np.isnan(camarilla_pivot[i]) or np.isnan(chop_align[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation and chop regime (both from 1d, already aligned)
        vol_confirm = volume_spike_1d[i] if not np.isnan(volume_spike_1d[i]) else False
        in_chop_regime = chop_regime[i] if not np.isnan(chop_align[i]) else False
        
        if position == 0:  # Flat - look for entry
            # Long when price breaks above H3 with volume confirmation in choppy market
            if close[i] > camarilla_H3[i] and vol_confirm and in_chop_regime:
                position = 1
                signals[i] = 0.25
                entry_price[i] = close[i]
            # Short when price breaks below L3 with volume confirmation in choppy market
            elif close[i] < camarilla_L3[i] and vol_confirm and in_chop_regime:
                position = -1
                signals[i] = -0.25
                entry_price[i] = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            # Exit when price returns to pivot level or breaks below L3 with volume
            exit_condition = (close[i] < camarilla_pivot[i]) or \
                           (close[i] < camarilla_L3[i] and vol_confirm)
            if exit_condition:
                position = 0
                signals[i] = 0.0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            # Exit when price returns to pivot level or breaks above H3 with volume
            exit_condition = (close[i] > camarilla_pivot[i]) or \
                           (close[i] > camarilla_H3[i] and vol_confirm)
            if exit_condition:
                position = 0
                signals[i] = 0.0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
    
    return signals