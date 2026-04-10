#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w HMA trend filter and volume confirmation
# - Long when price breaks above Donchian upper band (20-period high) AND 1w HMA(21) is rising AND volume > 1.5x 20-period average
# - Short when price breaks below Donchian lower band (20-period low) AND 1w HMA(21) is falling AND volume > 1.5x 20-period average
# - Exit when price returns to Donchian midpoint (mean reversion)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Donchian breakouts capture strong trends, HMA filter ensures we only trade with higher timeframe trend
# - Volume confirmation reduces false breakouts
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)

name = "1d_donchian_hma_volume_v1"
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
    
    # Pre-compute indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    upper_band = rolling_max(high, 20)
    lower_band = rolling_min(low, 20)
    midpoint = (upper_band + lower_band) / 2
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # 1w HMA(21) for trend filter
    def wma(arr, window):
        weights = np.arange(1, window + 1)
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.dot(arr[i - window + 1:i + 1], weights) / weights.sum()
        return result
    
    def hma(arr, window):
        half_len = window // 2
        sqrt_len = int(np.sqrt(window))
        wma_half = wma(arr, half_len)
        wma_full = wma(arr, window)
        raw_hma = 2 * wma_half - wma_full
        return wma(raw_hma, sqrt_len)
    
    close_1w = df_1w['close'].values
    hma_1w = hma(close_1w, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # HMA slope (rising/falling)
    hma_slope = np.zeros_like(hma_1w_aligned)
    hma_slope[1:] = np.diff(hma_1w_aligned)
    hma_rising = hma_slope > 0
    hma_falling = hma_slope < 0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(midpoint[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(hma_1w_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above upper band AND HMA rising AND volume spike
            if (close[i] > upper_band[i] and 
                hma_rising[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below lower band AND HMA falling AND volume spike
            elif (close[i] < lower_band[i] and 
                  hma_falling[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to midpoint (mean reversion)
            exit_long = (position == 1 and close[i] < midpoint[i])
            exit_short = (position == -1 and close[i] > midpoint[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals