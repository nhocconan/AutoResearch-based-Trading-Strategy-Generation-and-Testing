#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume spike and chop regime filter
# - Long when price breaks above Camarilla H3 level AND 1d volume > 1.5x 20-period average AND chop > 61.8 (ranging market)
# - Short when price breaks below Camarilla L3 level AND 1d volume > 1.5x 20-period average AND chop > 61.8
# - Exit when price returns to Camarilla H4/L4 levels (mean reversion in range)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Camarilla pivots work well in ranging markets which dominate BTC/ETH in 2025+
# - Volume confirmation reduces false breakouts
# - Chop filter ensures we only trade in ranging conditions where mean reversion works

name = "12h_1d_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 12h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 12h Camarilla pivot levels (based on previous day)
    # H4 = close + 1.5*(high-low), H3 = close + 1.0*(high-low), etc.
    # We use 1d data to calculate pivots for 12h timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous 1d bar
    camarilla_h4 = np.zeros_like(close_1d)
    camarilla_h3 = np.zeros_like(close_1d)
    camarilla_l3 = np.zeros_like(close_1d)
    camarilla_l4 = np.zeros_like(close_1d)
    
    for i in range(1, len(close_1d)):
        diff = high_1d[i-1] - low_1d[i-1]
        camarilla_h4[i] = close_1d[i-1] + 1.5 * diff
        camarilla_h3[i] = close_1d[i-1] + 1.0 * diff
        camarilla_l3[i] = close_1d[i-1] - 1.0 * diff
        camarilla_l4[i] = close_1d[i-1] - 1.5 * diff
    
    # Align Camarilla levels to 12h timeframe (1d values apply to following 12h bars)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Pre-compute 12h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 12h Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(TR over n) / (n * (max(high) - min(low)))) / log10(n)
    # CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    tr = np.zeros_like(close)
    tr[0] = high[0] - low[0]  # First bar TR
    for i in range(1, len(close)):
        tr[i] = true_range(high[i], low[i], close[i-1])
    
    # Rolling sum of TR
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Rolling max(high) and min(low)
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = np.zeros_like(close)
    for i in range(13, len(close)):
        if max_high[i] > min_low[i]:  # Avoid division by zero
            chop[i] = 100 * np.log10(tr_sum[i] / (14 * (max_high[i] - min_low[i]))) / np.log10(14)
        else:
            chop[i] = 50  # Neutral when no range
    
    chop_regime = chop > 61.8  # Ranging market
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop_regime[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above H3 AND volume spike AND ranging market
            if (close[i] > camarilla_h3_aligned[i] and 
                volume_spike[i] and 
                chop_regime[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below L3 AND volume spike AND ranging market
            elif (close[i] < camarilla_l3_aligned[i] and 
                  volume_spike[i] and 
                  chop_regime[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to H4/L4 levels (mean reversion)
            exit_long = (position == 1 and close[i] < camarilla_h4_aligned[i])
            exit_short = (position == -1 and close[i] > camarilla_l4_aligned[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals