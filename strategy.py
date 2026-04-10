#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot long/short with 1d volume spike and chop regime filter
# - Long when price touches Camarilla L3 support AND 1d volume > 2.0x 20-period volume SMA AND chop > 61.8 (range regime)
# - Short when price touches Camarilla H3 resistance AND 1d volume > 2.0x 20-period volume SMA AND chop > 61.8 (range regime)
# - Exit: opposite Camarilla touch or volume drops below average or chop < 38.2 (trend regime)
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 12-37 trades/year on 12h timeframe to stay within fee drag limits
# - Camarilla levels calculated from prior 1d bar (HLC) for look-ahead safety

name = "12h_1d_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 12h volume SMA for regime filter
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h Choppiness Index (CHOP) for regime detection
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low) over 14 periods)
    tr1 = pd.Series(high).rolling(2).max() - pd.Series(low).rolling(2).min()
    tr2 = abs(pd.Series(high).rolling(2).max() - pd.Series(close).shift(1).rolling(2).min())
    tr3 = abs(pd.Series(low).rolling(2).min() - pd.Series(close).shift(1).rolling(2).min())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr_14 / np.log10(highest_high_14 - lowest_low_14 + 1e-10))
    # Handle division by zero or invalid values
    chop = np.where((highest_high_14 - lowest_low_14) > 0, chop, 50.0)
    chop = np.where(np.isnan(chop), 50.0, chop)
    
    # Calculate 1d Camarilla levels from prior 1d bar (HLC) - aligned to 12h bars
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Typical price for prior day
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    camarilla_h3 = close_1d + (range_1d * 1.1 / 4.0)  # H3 = Close + Range * 1.1/4
    camarilla_l3 = close_1d - (range_1d * 1.1 / 4.0)  # L3 = Close - Range * 1.1/4
    camarilla_h4 = close_1d + (range_1d * 1.1 / 2.0)  # H4 = Close + Range * 1.1/2
    camarilla_l4 = close_1d - (range_1d * 1.1 / 2.0)  # L4 = Close - Range * 1.1/2
    
    # Align Camarilla levels to 12h timeframe (use prior day's levels for current bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate 1d volume for confirmation (prior day's volume)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    for i in range(40, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(chop[i]) or np.isnan(volume_sma_20[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 12h volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > 2.0 * volume_sma_20[i]
        
        # 1d volume confirmation: prior day volume > 2.0x its 20-period SMA
        vol_confirm_1d = volume_1d[i//16] > 2.0 * volume_sma_20_1d_aligned[i] if i//16 < len(volume_1d) else False
        
        # Regime filter: chop > 61.8 (range-bound market)
        chop_range = chop[i] > 61.8
        
        # Camarilla touch conditions (using prior bar's levels to avoid look-ahead)
        touch_h3 = high[i] >= camarilla_h3_aligned[i-1]  # High touches or exceeds H3
        touch_l3 = low[i] <= camarilla_l3_aligned[i-1]   # Low touches or goes below L3
        
        # Exit conditions
        exit_long = touch_h3 or not vol_confirm or not chop_range
        exit_short = touch_l3 or not vol_confirm or not chop_range
        
        if position == 0:  # Flat - look for entry
            if touch_l3 and vol_confirm and vol_confirm_1d and chop_range:
                position = -1  # Short at L3 resistance
                signals[i] = -0.25
            elif touch_h3 and vol_confirm and vol_confirm_1d and chop_range:
                position = 1   # Long at H3 support
                signals[i] = 0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals