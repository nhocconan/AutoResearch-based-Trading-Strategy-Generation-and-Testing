#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d volume spike and choppiness filter
# - Long: price breaks above Donchian(20) high AND 1d volume > 2.0x 20-period average AND chop < 61.8 (trending)
# - Short: price breaks below Donchian(20) low AND 1d volume > 2.0x 20-period average AND chop < 61.8 (trending)
# - Uses ATR(14) trailing stop: exit long if price < highest_high - 2.5*ATR, exit short if price < lowest_low + 2.5*ATR
# - Position size: ±0.25 (discrete to minimize fee churn)
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Donchian provides objective breakout levels
# - 1d volume spike confirms institutional participation
# - Choppiness filter avoids whipsaws in ranging markets
# - ATR stop manages risk without look-ahead

name = "12h_donchian_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d volume confirmation (20-period average)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute 12h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute ATR(14) for trailing stop
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 12h choppiness index (14-period)
    # Chop = 100 * log10(sum(atr,14) / (log10(highest_high - lowest_low,14))) / log10(14)
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hhll = highest_high - lowest_low
    # Avoid division by zero and log of zero
    hhll_safe = np.where(hhll > 0, hhll, 1e-10)
    chop = 100 * np.log10(atr_sum) / np.log10(14) / np.log10(hhll_safe)
    # Handle invalid values
    chop = np.where((atr_sum > 0) & (hhll > 0), chop, 50.0)  # default to neutral
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(volume_sma_20_aligned[i]) or np.isnan(atr[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        volume_current = volume[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i-1]  # break above previous period's high
        breakout_down = close[i] < lowest_low[i-1]  # break below previous period's low
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = volume_current > 2.0 * volume_sma_20_aligned[i]
        
        # Choppiness filter: chop < 61.8 indicates trending market (good for breakouts)
        chop_filter = chop[i] < 61.8
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: bullish breakout + volume confirmation + trending market
        if breakout_up and vol_confirm and chop_filter:
            enter_long = True
        
        # Short: bearish breakout + volume confirmation + trending market
        if breakout_down and vol_confirm and chop_filter:
            enter_short = True
        
        # Exit conditions: ATR trailing stop
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price drops below highest_high - 2.5*ATR
            exit_long = close[i] < (highest_high[i] - 2.5 * atr[i])
        elif position == -1:
            # Exit short if price rises above lowest_low + 2.5*ATR
            exit_short = close[i] > (lowest_low[i] + 2.5 * atr[i])
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals