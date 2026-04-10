#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with daily trend filter and volume confirmation
# - Long when price breaks above Camarilla H3 level AND 1d EMA50 > EMA200 (bullish trend)
# - Short when price breaks below Camarilla L3 level AND 1d EMA50 < EMA200 (bearish trend)
# - Volume confirmation: 4h volume > 1.3x 20-period 4h volume SMA
# - Exit: Price returns to Camarilla Pivot level or opposite breakout with volume
# - Position sizing: 0.25 discrete level
# - Uses 1d EMA trend filter to avoid counter-trend trades in choppy markets
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)

name = "4h_1d_camarilla_ema_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 and EMA200 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d EMA to 4h timeframe (completed 1d bar only)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Camarilla: Pivot = (H+L+C)/3, Range = H-L
    # H4 = Pivot + 1.1*Range/2, L4 = Pivot - 1.1*Range/2
    # H3 = Pivot + 1.1*Range/4, L3 = Pivot - 1.1*Range/4
    # H2 = Pivot + 1.1*Range/6, L2 = Pivot - 1.1*Range/6
    # H1 = Pivot + 1.1*Range/12, L1 = Pivot - 1.1*Range/12
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    h3_1d = pivot_1d + 1.1 * range_1d / 4.0
    l3_1d = pivot_1d - 1.1 * range_1d / 4.0
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    # Calculate 20-period volume SMA for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Track entry price for reference
    entry_price = np.full(n, np.nan)
    
    # Warmup period: need enough data for 1d EMA200
    warmup = max(200, 20)  # 1d EMA200 needs 200 days, plus 20 for volume SMA
    
    for i in range(warmup, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or
            np.isnan(pivot_1d_aligned[i]) or np.isnan(h3_1d_aligned[i]) or 
            np.isnan(l3_1d_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1d EMA50 > EMA200 for bullish, < for bearish
        bullish_trend = ema50_1d_aligned[i] > ema200_1d_aligned[i]
        bearish_trend = ema50_1d_aligned[i] < ema200_1d_aligned[i]
        
        # Volume confirmation: 4h volume > 1.3x 20-period volume SMA
        vol_confirm = volume[i] > 1.3 * volume_sma_20[i]
        
        # Camarilla breakout signals
        breakout_up = close[i] > h3_1d_aligned[i]   # Break above H3
        breakout_down = close[i] < l3_1d_aligned[i]  # Break below L3
        
        if position == 0:  # Flat - look for entry
            if breakout_up and bullish_trend and vol_confirm:
                position = 1
                signals[i] = 0.25
                entry_price[i] = close[i]
            elif breakout_down and bearish_trend and vol_confirm:
                position = -1
                signals[i] = -0.25
                entry_price[i] = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            # Exit on return to pivot or opposite breakout with volume
            exit_condition = (close[i] < pivot_1d_aligned[i]) or \
                           (breakout_down and vol_confirm)
            if exit_condition:
                position = 0
                signals[i] = 0.0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            # Exit on return to pivot or opposite breakout with volume
            exit_condition = (close[i] > pivot_1d_aligned[i]) or \
                           (breakout_up and vol_confirm)
            if exit_condition:
                position = 0
                signals[i] = 0.0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
    
    return signals