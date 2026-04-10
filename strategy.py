#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and choppiness regime filter
# - Long when price breaks above Camarilla H3 level AND 1d volume > 2.0x 20-period average volume AND 1d choppiness index > 61.8 (range regime)
# - Short when price breaks below Camarilla L3 level AND 1d volume > 2.0x 20-period average volume AND 1d choppiness index > 61.8 (range regime)
# - Exit when price returns to Camarilla Pivot point (mean reversion to equilibrium)
# - Uses discrete position sizing 0.30 to balance return and drawdown
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# - Camarilla levels provide high-probability reversal/breakout zones in ranging markets
# - Volume spike confirms institutional participation
# - Choppiness filter ensures we only trade in ranging markets where mean reversion works

name = "4h_1d_camarilla_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Pre-compute 1d typical price for Camarilla calculation
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    typical_price_1d = typical_price_1d.values
    
    # Pre-compute 1d Camarilla levels (based on previous day)
    # Camarilla formulas: H4 = PP + 1.1*(HL/2), H3 = PP + 1.1*(HL/4), L3 = PP - 1.1*(HL/4), L4 = PP - 1.1*(HL/2)
    # where PP = (H+L+C)/3, HL = H-L
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shifted by 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan  # First day has no previous
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels for previous day
    prev_pp = (prev_high + prev_low + prev_close) / 3
    prev_hl = prev_high - prev_low
    
    # Camarilla levels
    camarilla_h3 = prev_pp + 1.1 * prev_hl / 4
    camarilla_l3 = prev_pp - 1.1 * prev_hl / 4
    camarilla_pivot = prev_pp
    
    # Pre-compute 1d volume confirmation (20-period average)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_1d > (2.0 * vol_ma_1d)
    
    # Pre-compute 1d choppiness index (CHOP) - range detection
    # CHOP = 100 * LOG10(SUM(TR,14) / (MAX(HIGH,14) - MIN(LOW,14))) / LOG10(14)
    # Values > 61.8 indicate ranging market (good for mean reversion)
    # Values < 38.2 indicate trending market
    def true_range(high_arr, low_arr, close_arr):
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr1[0] = 0
        tr2[0] = 0
        tr3[0] = 0
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    tr_1d = true_range(high_1d, low_1d, close_1d)
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denominator = max_high_14 - min_low_14
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)  # Avoid division by zero
    chop_1d = 100 * np.log10(atr_14_1d / chop_denominator) / np.log10(14)
    chop_1d = np.where(np.isnan(chop_1d), 50, chop_1d)  # Default to neutral if invalid
    chop_range_regime = chop_1d > 61.8  # Ranging market
    
    # Align HTF indicators to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    chop_range_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_range_regime)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(volume_spike_1d_aligned[i]) or 
            np.isnan(chop_range_regime_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Camarilla H3 AND volume spike AND ranging regime
            if (close[i] > camarilla_h3_aligned[i] and 
                volume_spike_1d_aligned[i] and 
                chop_range_regime_aligned[i]):
                position = 1
                signals[i] = 0.30
            # Short conditions: price breaks below Camarilla L3 AND volume spike AND ranging regime
            elif (close[i] < camarilla_l3_aligned[i] and 
                  volume_spike_1d_aligned[i] and 
                  chop_range_regime_aligned[i]):
                position = -1
                signals[i] = -0.30
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to Camarilla Pivot point (mean reversion)
            exit_long = (position == 1 and close[i] <= camarilla_pivot_aligned[i])
            exit_short = (position == -1 and close[i] >= camarilla_pivot_aligned[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.30
                else:
                    signals[i] = -0.30
    
    return signals