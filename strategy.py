#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot levels with weekly trend filter and volume confirmation
# - Long when price touches Camarilla L3 support AND weekly HMA(21) is rising AND volume > 1.5x 20-day average volume
# - Short when price touches Camarilla H3 resistance AND weekly HMA(21) is falling AND volume > 1.5x 20-day average volume
# - Exit when price crosses the Camarilla pivot point (midpoint between H3 and L3)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
# - Camarilla levels provide precise support/resistance with high probability reactions
# - Weekly HMA filter ensures we trade with the higher timeframe trend
# - Volume confirmation reduces false signals

name = "1d_1w_camarilla_hma_volume_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Pre-compute 1d OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 1d Camarilla levels (based on previous day's range)
    # Camarilla levels: H4, H3, H2, H1, Pivot, L1, L2, L3, L4
    # H3 = Close + 1.1 * (High - Low) / 2
    # L3 = Close - 1.1 * (High - Low) / 2
    # Pivot = (High + Low + Close) / 3
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # First bar uses current values
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3
    
    # Pre-compute 1d volume confirmation (20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute weekly HMA(21) for trend filter
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(arr, window):
        weights = np.arange(1, window + 1)
        return np.convolve(arr, weights / weights.sum(), mode='same')
    
    close_1w = df_1w['close'].values
    wma_half = wma(close_1w, half_len)
    wma_full = wma(close_1w, 21)
    hma_21 = wma(2 * wma_half - wma_full, sqrt_len)
    
    # HMA trend: rising when current HMA > previous HMA
    hma_rising = np.zeros_like(hma_21, dtype=bool)
    hma_rising[1:] = hma_21[1:] > hma_21[:-1]
    hma_falling = np.zeros_like(hma_21, dtype=bool)
    hma_falling[1:] = hma_21[1:] < hma_21[:-1]
    
    # Align HTF indicators to 1d timeframe
    hma_rising_aligned = align_htf_to_ltf(prices, df_1w, hma_rising)
    hma_falling_aligned = align_htf_to_ltf(prices, df_1w, hma_falling)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(camarilla_pivot[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(hma_rising_aligned[i]) or np.isnan(hma_falling_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price touches L3 support AND weekly HMA rising AND volume spike
            if (low[i] <= camarilla_l3[i] and 
                hma_rising_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price touches H3 resistance AND weekly HMA falling AND volume spike
            elif (high[i] >= camarilla_h3[i] and 
                  hma_falling_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses the Camarilla pivot point
            exit_long = (position == 1 and close[i] < camarilla_pivot[i])
            exit_short = (position == -1 and close[i] > camarilla_pivot[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals