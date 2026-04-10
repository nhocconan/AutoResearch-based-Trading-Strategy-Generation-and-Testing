#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with 1w HMA trend filter and volume confirmation
# - Long when price breaks above Camarilla H3 level AND 1w HMA(21) > 1w HMA(50) AND volume > 1.5x 20-period average volume
# - Short when price breaks below Camarilla L3 level AND 1w HMA(21) < 1w HMA(50) AND volume > 1.5x 20-period average volume
# - Exit when price crosses back inside Camarilla H3/L3 levels
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
# - Camarilla pivots identify key intraday support/resistance levels
# - HMA trend filter ensures we trade with the weekly trend
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
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute daily OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 1d Camarilla pivot levels (based on previous day)
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # H3 = close + 1.1*(high-low)/2
    # L3 = close - 1.1*(high-low)/2
    # We use previous day's range to calculate today's levels
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # First bar uses current values
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Pre-compute 1d volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1w HMA(21) and HMA(50) for trend filter
    # HMA formula: WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    def wma(arr, window):
        if window <= 1:
            return arr.copy()
        weights = np.arange(1, window + 1)
        return np.convolve(arr, weights/weights.sum(), mode='same')
    
    def hma(arr, window):
        half_window = window // 2
        sqrt_window = int(np.sqrt(window))
        wma_half = wma(arr, half_window)
        wma_full = wma(arr, window)
        raw_hma = 2 * wma_half - wma_full
        return wma(raw_hma, sqrt_window)
    
    close_1w = df_1w['close'].values
    hma_21_1w = hma(close_1w, 21)
    hma_50_1w = hma(close_1w, 50)
    
    # Align HTF indicators to 1d timeframe
    hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w)
    hma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_50_1w)
    
    # Trend filter: bullish when HMA(21) > HMA(50), bearish when HMA(21) < HMA(50)
    bullish_trend = hma_21_1w_aligned > hma_50_1w_aligned
    bearish_trend = hma_21_1w_aligned < hma_50_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(hma_21_1w_aligned[i]) or 
            np.isnan(hma_50_1w_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Camarilla H3 AND bullish trend AND volume spike
            if (close[i] > camarilla_h3[i] and 
                bullish_trend[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Camarilla L3 AND bearish trend AND volume spike
            elif (close[i] < camarilla_l3[i] and 
                  bearish_trend[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses back inside Camarilla H3/L3 levels
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