#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h volume spike and chop regime filter
# - Long when price breaks above Camarilla H3 level + 4h volume > 2.0x 20-period volume SMA + Chop(14) > 61.8 (range regime)
# - Short when price breaks below Camarilla L3 level + 4h volume > 2.0x 20-period volume SMA + Chop(14) > 61.8
# - Exit: price returns to Camarilla pivot point (mean reversion within range)
# - Position sizing: 0.20 discrete level
# - Camarilla levels derived from prior 4h bar provide structure in ranging markets
# - Volume confirms institutional participation, chop filter ensures we trade in ranging markets
# - Works in bull/bear: mean reversion at pivot levels works in all regimes, chop filter avoids strong trends
# - 1h timeframe targets 15-37 trades/year with strict entry conditions to minimize fee drag
# - Session filter (08-20 UTC) to reduce noise trades

name = "1h_4h_camarilla_vol_chop_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 4h Camarilla pivot levels (based on prior 4h bar)
    # Camarilla formula: H4 = close + 1.1*(high-low)*1.1/2, H3 = close + 1.1*(high-low)*1.1/4
    # L3 = close - 1.1*(high-low)*1.1/4, L4 = close - 1.1*(high-low)*1.1/2
    # Pivot = (high + low + close) / 3
    hl_range_4h = df_4h['high'].values - df_4h['low'].values
    camarilla_pivot = (df_4h['high'].values + df_4h['low'].values + df_4h['close'].values) / 3
    camarilla_h3 = camarilla_pivot + 1.1 * hl_range_4h * 1.1 / 4
    camarilla_l3 = camarilla_pivot - 1.1 * hl_range_4h * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe (using completed 4h bar)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pivot)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    
    # Calculate 4h Chopiness Index (14-period) for regime filter
    # True Range
    tr1 = np.maximum(high - low, 
                     np.maximum(np.abs(high - np.roll(close, 1)), 
                                np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]
    # Sum of TR over period
    sum_tr = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over period
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Chop formula: 100 * log10(sum_TR / (HH - LL)) / log10(N)
    # Avoid division by zero and log of zero/negative
    hl_range = hh - ll
    chop = np.where((hl_range > 0) & (sum_tr > 0), 
                    100 * np.log10(sum_tr / hl_range) / np.log10(14), 
                    50)  # default to neutral when invalid
    
    # Calculate 4h volume SMA(20) for confirmation
    volume_4h = df_4h['volume'].values
    volume_sma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_sma_20_4h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_pivot_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(chop[i]) or 
            np.isnan(volume_sma_20_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Get current 4h volume for volume spike confirmation
        vol_4h_current = align_htf_to_ltf(prices, df_4h, df_4h['volume'].values)
        
        # Volume confirmation: current 4h volume > 2.0x 20-period SMA (volume spike)
        vol_confirm = vol_4h_current[i] > 2.0 * volume_sma_20_4h_aligned[i]
        
        # Regime filter: Chop > 61.8 indicates ranging market (favorable for mean reversion at pivot levels)
        ranging_market = chop[i] > 61.8
        
        # Camarilla breakout signals
        breakout_up = close[i] > camarilla_h3_aligned[i]   # Price breaks above H3 level
        breakout_down = close[i] < camarilla_l3_aligned[i] # Price breaks below L3 level
        return_to_pivot = abs(close[i] - camarilla_pivot_aligned[i]) < (camarilla_h3_aligned[i] - camarilla_l3_aligned[i]) * 0.15  # Within 15% of pivot
        
        # Entry conditions: Camarilla breakout with volume and regime confirmation
        long_entry = breakout_up and vol_confirm and ranging_market
        short_entry = breakout_down and vol_confirm and ranging_market
        
        # Exit conditions: price returns to Camarilla pivot point (mean reversion)
        long_exit = return_to_pivot  # Exit long when price returns to pivot
        short_exit = return_to_pivot  # Exit short when price returns to pivot
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.20
            elif short_entry:
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if long_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        else:  # position == -1 (Short position) - look for exit
            if short_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
    
    return signals