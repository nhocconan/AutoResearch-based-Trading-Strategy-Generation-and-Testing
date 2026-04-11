#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R + Volume Spike + Chop Regime Filter
# - Williams %R(14): overbought > -20, oversold < -80
# - Long: Williams %R crosses above -80 from below (oversold bounce) AND volume > 2.0x 24-period average AND chop > 61.8 (range regime)
# - Short: Williams %R crosses below -20 from above (overbought rejection) AND volume > 2.0x 24-period average AND chop > 61.8 (range regime)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Williams %R identifies reversal points in ranging markets
# - Volume spike confirms conviction behind the move
# - Chop regime filter (EHLERS) ensures we only trade in ranging conditions where mean reversion works
# - Works in both bull (buy dips) and bear (sell rallies) markets during consolidation phases

name = "12h_williamsr_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d chop regime filter (Ehlers Choppy Index)
    hl_range_1d = pd.Series(df_1d['high'] - df_1d['low']).rolling(window=14, min_periods=14).sum()
    true_range_1d = pd.Series(np.maximum(df_1d['high'] - df_1d['low'], 
                                        np.maximum(np.abs(df_1d['high'] - df_1d['close'].shift(1)),
                                                  np.abs(df_1d['low'] - df_1d['close'].shift(1))))).rolling(window=14, min_periods=14).sum()
    chop_1d = 100 * np.log10(hl_range_1d / true_range_1d) / np.log10(14)
    chop_1d_values = chop_1d.fillna(50).values  # fill NaN with neutral 50
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d_values)
    
    # Pre-compute 1d volume confirmation (24-period average)
    volume_1d = df_1d['volume'].values
    volume_sma_24_1d = pd.Series(volume_1d).rolling(window=24, min_periods=24).mean().values
    volume_sma_24_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_24_1d)
    
    # Pre-compute Williams %R on 12h timeframe
    highest_high = pd.Series(close).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).fillna(-50).values  # neutral when undefined
    
    # Williams %R previous value for crossover detection
    williams_r_prev = np.roll(williams_r, 1)
    williams_r_prev[0] = williams_r[0]  # first value same as current
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(williams_r_prev[i]) or
            np.isnan(volume_sma_24_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        volume_current = volume[i]
        
        # Williams %R crossover signals
        williams_r_cross_up = (williams_r_prev[i] <= -80) and (williams_r[i] > -80)  # crossing above -80 (oversold bounce)
        williams_r_cross_down = (williams_r_prev[i] >= -20) and (williams_r[i] < -20)  # crossing below -20 (overbought rejection)
        
        # Volume confirmation: current volume > 2.0x 24-period average
        vol_confirm = volume_current > 2.0 * volume_sma_24_aligned[i]
        
        # Chop regime filter: only trade in ranging markets (chop > 61.8)
        chop_filter = chop_1d_aligned[i] > 61.8
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Williams %R crosses above -80 (oversold bounce) + volume confirmation + chop regime
        if williams_r_cross_up and vol_confirm and chop_filter:
            enter_long = True
        
        # Short: Williams %R crosses below -20 (overbought rejection) + volume confirmation + chop regime
        if williams_r_cross_down and vol_confirm and chop_filter:
            enter_short = True
        
        # Exit conditions: opposite Williams %R crossover or loss of regime
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Williams %R crosses above -20 (overbought) OR chop regime ends
            williams_r_cross_down_exit = (williams_r_prev[i] >= -20) and (williams_r[i] < -20)
            exit_long = williams_r_cross_down_exit or (chop_1d_aligned[i] <= 61.8)
        elif position == -1:
            # Exit short if Williams %R crosses below -80 (oversold) OR chop regime ends
            williams_r_cross_up_exit = (williams_r_prev[i] <= -80) and (williams_r[i] > -80)
            exit_short = williams_r_cross_up_exit or (chop_1d_aligned[i] <= 61.8)
        
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