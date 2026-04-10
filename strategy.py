#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with 1w HMA trend filter and volume confirmation
# - Long when price breaks above Camarilla H3 level AND 1w HMA(21) is rising AND volume > 1.5x 20-period average volume
# - Short when price breaks below Camarilla L3 level AND 1w HMA(21) is falling AND volume > 1.5x 20-period average volume
# - Exit when price crosses back inside the Camarilla H3-L3 range
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
# - Camarilla pivots identify key intraday support/resistance levels with high probability breakouts
# - 1w HMA filter ensures we trade with the weekly trend to avoid counter-trend whipsaws
# - Volume confirmation reduces false breakouts

name = "1d_1w_camarilla_hma_volume_v1"
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
    
    # Pre-compute 1d Camarilla pivot levels (based on previous day)
    # Camarilla: H4 = close + 1.5*(high-low), H3 = close + 1.25*(high-low), etc.
    # We use H3/L3 for entries (stronger levels)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # First bar: use current values
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_range = prev_high - prev_low
    camarilla_h3 = prev_close + 1.25 * camarilla_range
    camarilla_l3 = prev_close - 1.25 * camarilla_range
    
    # Pre-compute 1d volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1w HMA(21) for trend filter
    # HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n))
    close_1w = df_1w['close'].values
    
    def wma(arr, window):
        """Weighted Moving Average"""
        if len(arr) < window:
            return np.full_like(arr, np.nan, dtype=float)
        weights = np.arange(1, window + 1)
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.dot(arr[i - window + 1:i + 1], weights) / weights.sum()
        return result
    
    # Calculate HMA(21)
    half_window = 21 // 2
    sqrt_window = int(np.sqrt(21))
    
    wma_half = wma(close_1w, half_window)
    wma_full = wma(close_1w, 21)
    hma_input = 2 * wma_half - wma_full
    hma_1w = wma(hma_input, sqrt_window)
    
    # HMA is rising when current > previous, falling when current < previous
    hma_rising = np.zeros_like(hma_1w, dtype=bool)
    hma_falling = np.zeros_like(hma_1w, dtype=bool)
    hma_rising[1:] = hma_1w[1:] > hma_1w[:-1]
    hma_falling[1:] = hma_1w[1:] < hma_1w[:-1]
    
    # Align HTF indicators to 1d timeframe
    hma_rising_aligned = align_htf_to_ltf(prices, df_1w, hma_rising)
    hma_falling_aligned = align_htf_to_ltf(prices, df_1w, hma_falling)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(hma_rising_aligned[i]) or np.isnan(hma_falling_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above H3 AND HMA rising AND volume spike
            if (close[i] > camarilla_h3[i] and 
                hma_rising_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below L3 AND HMA falling AND volume spike
            elif (close[i] < camarilla_l3[i] and 
                  hma_falling_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses back inside the Camarilla H3-L3 range
            exit_long = (position == 1 and close[i] < camarilla_h3[i])
            exit_short = (position == -1 and close[i] > camarilla_l3[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals