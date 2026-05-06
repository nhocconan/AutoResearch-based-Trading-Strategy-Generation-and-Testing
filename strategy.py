#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Bollinger Band squeeze breakout with volume confirmation and ADX trend filter
# Long when price breaks above upper BB(20,2) AND BB width < 20th percentile (squeeze) AND ADX(14) > 20 (trending) AND volume > 1.5 * avg_volume(20)
# Short when price breaks below lower BB(20,2) AND BB width < 20th percentile AND ADX(14) > 20 AND volume > 1.5 * avg_volume(20)
# Exit when price returns to middle BB(20) or BB width > 50th percentile (squeeze end)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Bollinger squeeze breakouts capture low-volatility breakouts with high follow-through
# Volume confirmation ensures institutional participation
# ADX filter avoids false breakouts in ranging markets
# Works in both bull (breakout continuations) and bear (breakdown continuations) markets

name = "4h_1dBB_Squeeze_Breakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Bollinger Bands and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for BB and ADX
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Bollinger Bands (20,2)
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    middle_bb = sma_20
    
    # Calculate BB width: (Upper - Lower) / Middle
    bb_width = (upper_bb - lower_bb) / middle_bb
    # Handle division by zero
    bb_width = np.where(middle_bb == 0, 0, bb_width)
    
    # Calculate BB width percentiles for squeeze detection (20th percentile = squeeze)
    bb_width_series = pd.Series(bb_width)
    bb_width_20th = bb_width_series.rolling(window=50, min_periods=20).quantile(0.20).values
    bb_width_50th = bb_width_series.rolling(window=50, min_periods=20).quantile(0.50).values
    
    # Calculate 1d ADX(14)
    # ADX requires +DI, -DI, and TR
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Wilder's smoothing (equivalent to EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 4h timeframe (wait for completed 1d bar)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    middle_bb_aligned = align_htf_to_ltf(prices, df_1d, middle_bb)
    bb_width_20th_aligned = align_htf_to_ltf(prices, df_1d, bb_width_20th)
    bb_width_50th_aligned = align_htf_to_ltf(prices, df_1d, bb_width_50th)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(middle_bb_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper BB AND BB width < 20th percentile (squeeze) AND ADX > 20 AND volume spike
            if (close[i] > upper_bb_aligned[i] and 
                bb_width_20th_aligned[i] > 0 and  # Valid percentile
                (upper_bb_aligned[i] - lower_bb_aligned[i]) / middle_bb_aligned[i] < bb_width_20th_aligned[i] and
                adx_aligned[i] > 20 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower BB AND BB width < 20th percentile (squeeze) AND ADX > 20 AND volume spike
            elif (close[i] < lower_bb_aligned[i] and 
                  bb_width_20th_aligned[i] > 0 and  # Valid percentile
                  (upper_bb_aligned[i] - lower_bb_aligned[i]) / middle_bb_aligned[i] < bb_width_20th_aligned[i] and
                  adx_aligned[i] > 20 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price returns to middle BB OR BB width > 50th percentile (squeeze end)
            if (close[i] <= middle_bb_aligned[i] or 
                (upper_bb_aligned[i] - lower_bb_aligned[i]) / middle_bb_aligned[i] > bb_width_50th_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price returns to middle BB OR BB width > 50th percentile (squeeze end)
            if (close[i] >= middle_bb_aligned[i] or 
                (upper_bb_aligned[i] - lower_bb_aligned[i]) / middle_bb_aligned[i] > bb_width_50th_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals