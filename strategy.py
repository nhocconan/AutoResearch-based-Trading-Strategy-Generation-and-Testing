#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h HMA21 trend filter and volume confirmation.
# Enter long when price breaks above 4h Donchian upper band (20-period high), 12h HMA21 trending up, and volume > 1.5x 20-bar average.
# Enter short when price breaks below 4h Donchian lower band (20-period low), 12h HMA21 trending down, and volume > 1.5x 20-bar average.
# Exit when price crosses the 12h HMA21 in the opposite direction.
# Uses discrete position sizing (0.25) to limit drawdown and fee drag.
# Target: 80-150 total trades over 4 years (20-38/year) to avoid excessive fee churn.
# Donchian channels provide clear trend-following breakouts; HMA21 filters for smooth 12h trend;
# Volume confirmation ensures breakouts have participation.

name = "4h_DonchianBreakout_12hHMA21_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for HMA21 trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h HMA21
    close_12h = df_12h['close'].values
    # HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n)
    n_hma = 21
    half_n = n_hma // 2
    sqrt_n = int(np.sqrt(n_hma))
    
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights/weights.sum(), mode='valid')
    
    # Pad to handle convolution
    wma_half = np.full_like(close_12h, np.nan)
    wma_full = np.full_like(close_12h, np.nan)
    
    if len(close_12h) >= half_n:
        wma_half[half_n-1:] = wma(close_12h, half_n)
    if len(close_12h) >= n_hma:
        wma_full[n_hma-1:] = wma(close_12h, n_hma)
    
    raw_hma = 2 * wma_half - wma_full
    hma_21 = np.full_like(raw_hma, np.nan)
    if len(raw_hma) >= sqrt_n:
        hma_21[sqrt_n-1:] = wma(raw_hma[sqrt_n-1:], sqrt_n)
    
    hma_21_aligned = align_htf_to_ltf(prices, df_12h, hma_21)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_window = 20
    upper_band = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_band = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donchian_window, 20, 30)  # Ensure sufficient history
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(hma_21_aligned[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 12h HMA21 trend: slope over 3 periods
        if i >= 3:
            hma_slope = (hma_21_aligned[i] - hma_21_aligned[i-3]) / 3
            hma_trend_up = hma_slope > 0
            hma_trend_down = hma_slope < 0
        else:
            hma_trend_up = False
            hma_trend_down = False
        
        # Donchian breakout conditions
        breakout_up = close[i] > upper_band[i]
        breakout_down = close[i] < lower_band[i]
        
        # Exit condition: price crosses 12h HMA21 in opposite direction
        exit_long = close[i] < hma_21_aligned[i]
        exit_short = close[i] > hma_21_aligned[i]
        
        # Handle entries and exits
        if breakout_up and hma_trend_up and vol_confirm and position <= 0:
            signals[i] = 0.25
            position = 1
        elif breakout_down and hma_trend_down and vol_confirm and position >= 0:
            signals[i] = -0.25
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals