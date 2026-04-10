#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout + 12h volume confirmation + chop regime filter
# - Long when price breaks above Camarilla H3 (1d) AND 12h volume > 1.5x 20-period average AND chop > 61.8 (trending)
# - Short when price breaks below Camarilla L3 (1d) AND 12h volume > 1.5x 20-period average AND chop > 61.8 (trending)
# - Exit when price crosses Camarilla pivot point (PP) or opposite breakout occurs
# - Uses discrete position sizing 0.25 to limit fee churn
# - Camarilla levels from 1d provide institutional support/resistance
# - Volume confirmation reduces false breakouts
# - Chop filter ensures we trade in trending regimes only (avoids whipsaws in ranging markets)
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)

name = "4h_12h_camarilla_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    if len(df_1d) < 1 or len(df_12h) < 20:
        return np.zeros(n)
    
    # Pre-compute 4h price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute 12h volume confirmation
    vol_ma_12h = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_spike_12h = df_12h['volume'].values > (1.5 * vol_ma_12h)
    
    # Pre-compute 1d Camarilla levels (based on previous day's OHLC)
    # Camarilla formulas:
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.0 * (high - low)
    # H2 = close + 0.75 * (high - low)
    # H1 = close + 0.5 * (high - low)
    # L1 = close - 0.5 * (high - low)
    # L2 = close - 0.75 * (high - low)
    # L3 = close - 1.0 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # PP = (high + low + close) / 3
    
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    camarilla_h3 = prev_close + 1.0 * (prev_high - prev_low)
    camarilla_l3 = prev_close - 1.0 * (prev_high - prev_low)
    camarilla_pp = (prev_high + prev_low + prev_close) / 3.0
    
    # Pre-compute 4h Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(14)
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    # We want trending markets: CHOP < 38.2
    def true_range(high, low, close_prev):
        tr1 = high - low
        tr2 = np.abs(high - close_prev)
        tr3 = np.abs(low - close_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR(14)
    tr = np.zeros(len(close))
    tr[0] = high[0] - low[0]  # First bar
    for i in range(1, len(close)):
        tr[i] = true_range(high[i], low[i], close[i-1])
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate highest high and lowest low over 14 periods
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop_raw = np.zeros(len(close))
    for i in range(14, len(close)):
        if atr_14[i] > 0 and (highest_high_14[i] - lowest_low_14[i]) > 0:
            sum_atr = np.sum(atr_14[i-13:i+1])  # Sum of last 14 ATR values
            chop_raw[i] = 100 * np.log10(sum_atr) / np.log10(14) / np.log10(highest_high_14[i] - lowest_low_14[i])
        else:
            chop_raw[i] = 50.0  # Neutral value
    
    # Chop filter: we want trending markets (CHOP < 38.2)
    chop_filter = chop_raw < 38.2
    
    # Align HTF indicators to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    volume_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_spike_12h)
    chop_filter_aligned = align_htf_to_ltf(prices, df_12h, chop_filter)  # Using 12h for chop alignment
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(volume_spike_12h_aligned[i]) or 
            np.isnan(chop_filter_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Camarilla H3 AND 12h volume spike AND trending regime (chop < 38.2)
            if (close[i] > camarilla_h3_aligned[i] and 
                volume_spike_12h_aligned[i] and 
                chop_filter_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Camarilla L3 AND 12h volume spike AND trending regime (chop < 38.2)
            elif (close[i] < camarilla_l3_aligned[i] and 
                  volume_spike_12h_aligned[i] and 
                  chop_filter_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses Camarilla pivot point OR opposite breakout occurs
            exit_long = (position == 1 and 
                        (close[i] < camarilla_pp_aligned[i] or close[i] < camarilla_l3_aligned[i]))
            exit_short = (position == -1 and 
                         (close[i] > camarilla_pp_aligned[i] or close[i] > camarilla_h3_aligned[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals