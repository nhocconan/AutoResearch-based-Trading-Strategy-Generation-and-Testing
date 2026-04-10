#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + HMA(21) trend + volume confirmation
# - Primary: 4h timeframe for balanced trade frequency and reduced fee drag
# - HTF: 12h for trend confirmation (HMA direction)
# - Long: Price breaks above Donchian(20) high + 12h HMA(21) rising + volume > 1.5x 20-period MA
# - Short: Price breaks below Donchian(20) low + 12h HMA(21) falling + volume > 1.5x 20-period MA
# - Exit: Price reverts to Donchian midpoint or ATR-based trailing stop (signal=0)
# - Position sizing: 0.25 (discrete level)
# - Target: 75-200 total trades over 4 years (19-50/year) - within 4h sweet spot
# - Works in bull/bear: Donchian breakouts capture trends, volume confirmation filters false signals, 12h HMA avoids counter-trend trades

name = "4h_12h_donchian_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h OHLCV
    open_4h = prices['open'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    volume_4h = prices['volume'].values
    
    # Pre-compute 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 4h Donchian Channel (20-period)
    highest_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_20 + lowest_20) / 2.0
    
    # Calculate 12h HMA(21) for trend confirmation
    # HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n)
    def wma(values, period):
        if len(values) < period:
            return np.full_like(values, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(values, weights / weights.sum(), mode='valid')
    
    def hma(values, period):
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        if len(values) < period:
            return np.full_like(values, np.nan)
        wma_half = wma(values, half_period)
        wma_full = wma(values, period)
        # Align arrays: wma_half starts at index half_period-1, wma_full at period-1
        raw_hma = 2 * wma_half[-len(wma_full):] - wma_full
        hma_values = wma(raw_hma, sqrt_period)
        # Pad with NaN to match original length
        result = np.full_like(values, np.nan)
        result[period-1:] = hma_values
        return result
    
    hma_12h = hma(close_12h, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 4h volume moving average (20-period) for volume confirmation
    volume_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(hma_12h_aligned[i]) or 
            np.isnan(volume_ma_20_4h[i])):
            signals[i] = 0.0
            continue
        
        # Trend condition: 12h HMA direction (rising/falling)
        if i >= 21:  # Need previous HMA value for slope
            hma_rising = hma_12h_aligned[i] > hma_12h_aligned[i-1]
            hma_falling = hma_12h_aligned[i] < hma_12h_aligned[i-1]
        else:
            hma_rising = False
            hma_falling = False
        
        # Volume confirmation: current volume > 1.5x 20-period MA
        volume_spike = volume_4h[i] > 1.5 * volume_ma_20_4h[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian high + HMA rising + volume spike
            if (close_4h[i] > highest_20[i] and hma_rising and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian low + HMA falling + volume spike
            elif (close_4h[i] < lowest_20[i] and hma_falling and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price reverts to Donchian midpoint (mean reversion)
            # 2. Opposite Donchian breakout with volume (stop and reverse)
            
            if position == 1:  # Long position
                exit_condition = (
                    close_4h[i] < donchian_mid[i] or  # Reverted to midpoint
                    (close_4h[i] < lowest_20[i] and volume_spike)  # Break below low with volume
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = (
                    close_4h[i] > donchian_mid[i] or  # Reverted to midpoint
                    (close_4h[i] > highest_20[i] and volume_spike)  # Break above high with volume
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals