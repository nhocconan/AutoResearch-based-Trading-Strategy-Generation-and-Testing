#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation
# - Donchian(20) breakout on 1d: long when price > upper band, short when price < lower band
# - 1w HMA(21) trend filter: only take longs when price > HMA, shorts when price < HMA
# - Volume confirmation: current volume > 1.5x 20-period average on 1d
# - ATR-based stoploss: exit when price moves against position by 2.5 * ATR(14)
# - Discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 7-25 trades/year (30-100 total over 4 years) to stay within fee drag limits
# - Donchian breakouts capture strong trending moves
# - HMA filter ensures we trade with the higher timeframe trend
# - Volume confirmation filters out weak breakouts
# - Works in both bull (breakouts with volume) and bear (breakdowns with volume) markets

name = "1d_donchian_20w_hma_volume_v1"
timeframe = "1d"
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
    entry_price = 0.0
    
    # Load 1d data ONCE before loop for Donchian and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Pre-compute 1d volume confirmation (20-period average)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute 1d ATR(14) for stoploss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close, 1))
    tr3 = np.abs(low_1d - np.roll(close, 1))
    tr2[0] = tr1[0]  # first bar has no previous close
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Load 1w data ONCE before loop for HMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # Pre-compute 1w HMA(21)
    close_1w = df_1w['close'].values
    half_length = int(21 / 2) + 1
    wma1 = pd.Series(close_1w).rolling(window=half_length, min_periods=half_length).mean().values
    wma2 = pd.Series(close_1w).rolling(window=21, min_periods=21).mean().values
    raw_hma = 2 * wma1 - wma2
    hma_21_1w = pd.Series(raw_hma).rolling(window=int(np.sqrt(21)), min_periods=int(np.sqrt(21))).mean().values
    hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w)
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(volume_sma_20_aligned[i]) or np.isnan(atr_14_aligned[i]) or
            np.isnan(hma_21_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Donchian breakout conditions
        breakout_up = close_price > donchian_upper_aligned[i]
        breakout_down = close_price < donchian_lower_aligned[i]
        
        # 1w HMA trend filter
        above_hma = close_price > hma_21_aligned[i]
        below_hma = close_price < hma_21_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # ATR stoploss level
        atr_value = atr_14_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Donchian breakout up + above 1w HMA + volume confirmation
        if breakout_up and above_hma and vol_confirm:
            enter_long = True
        
        # Short: Donchian breakout down + below 1w HMA + volume confirmation
        if breakout_down and below_hma and vol_confirm:
            enter_short = True
        
        # Exit conditions: ATR stoploss or Donchian breakout in opposite direction
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if stoploss hit OR Donchian breakout down
            if close_price < entry_price - 2.5 * atr_value:
                exit_long = True
            elif breakout_down:
                exit_long = True
        elif position == -1:
            # Exit short if stoploss hit OR Donchian breakout up
            if close_price > entry_price + 2.5 * atr_value:
                exit_short = True
            elif breakout_up:
                exit_short = True
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            entry_price = close_price
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            entry_price = close_price
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