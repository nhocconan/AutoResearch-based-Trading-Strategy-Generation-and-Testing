#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout + 1d volume confirmation + chop regime filter
# - Long when price breaks above Camarilla H3 level AND 1d volume > 1.8x 20-period average AND chop < 61.8 (trending)
# - Short when price breaks below Camarilla L3 level AND 1d volume > 1.8x 20-period average AND chop < 61.8 (trending)
# - Exit when price crosses Camarilla H4/L4 levels (strong reversal) or chop > 61.8 (range)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Camarilla pivots from 1d provide institutional support/resistance levels
# - Volume confirmation ensures breakout validity
# - Chop filter avoids false signals in ranging markets

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
    
    # Pre-compute 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 12h chop regime (14-period)
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate True Range for chop
    close_prev = np.roll(close, 1)
    close_prev[0] = close[0]
    tr = true_range(high, low, close_prev)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate chop: log(sum(ATR)/log(n)*max(high-low)) * 100
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    hhll = pd.Series(high - low).rolling(window=14, min_periods=14).max().values
    chop = np.where((atr_sum > 0) & (hhll > 0), 
                    np.log10(atr_sum / hhll) / np.log10(14) * 100, 50)
    
    # Pre-compute 12h volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Pre-compute 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels calculation
    # H4 = close + 1.5*(high-low)*1.1/2
    # H3 = close + 1.25*(high-low)*1.1/2
    # L3 = close - 1.25*(high-low)*1.1/2
    # L4 = close - 1.5*(high-low)*1.1/2
    hl_range_1d = high_1d - low_1d
    camarilla_h4 = close_1d + (1.5 * hl_range_1d * 1.1 / 2)
    camarilla_h3 = close_1d + (1.25 * hl_range_1d * 1.1 / 2)
    camarilla_l3 = close_1d - (1.25 * hl_range_1d * 1.1 / 2)
    camarilla_l4 = close_1d - (1.5 * hl_range_1d * 1.1 / 2)
    
    # Align HTF indicators to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(chop[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Camarilla H3 AND volume spike AND chop < 61.8 (trending)
            if (close[i] > camarilla_h3_aligned[i] and 
                volume_spike[i] and 
                chop[i] < 61.8):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Camarilla L3 AND volume spike AND chop < 61.8 (trending)
            elif (close[i] < camarilla_l3_aligned[i] and 
                  volume_spike[i] and 
                  chop[i] < 61.8):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses Camarilla H4/L4 OR chop > 61.8 (range)
            exit_long = (position == 1 and 
                        (close[i] < camarilla_l4_aligned[i] or chop[i] > 61.8))
            exit_short = (position == -1 and 
                         (close[i] > camarilla_h4_aligned[i] or chop[i] > 61.8))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals