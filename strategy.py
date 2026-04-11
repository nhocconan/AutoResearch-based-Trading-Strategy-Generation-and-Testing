#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume spike and weekly trend filter
# - Camarilla pivot levels from daily data provide high-probability support/resistance
# - Long when price breaks above H3 with 1d volume > 2x 20-period average
# - Short when price breaks below L3 with 1d volume > 2x 20-period average
# - Weekly EMA21 trend filter: only take longs above weekly EMA21, shorts below
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Camarilla pivots work in both bull and bear markets as reversal/continuation levels
# - Volume confirmation ensures genuine breakouts
# - Weekly trend filter aligns with higher timeframe momentum

name = "12h_camarilla_breakout_volume_trend_v1"
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
    
    # Load 1d data ONCE before loop for Camarilla pivots and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d volume confirmation (20-period average)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute Camarilla pivot levels from 1d data
    # Camarilla formulas: 
    # H4 = close + 1.1*(high-low)*1.1/2
    # H3 = close + 1.1*(high-low)*1.1/4
    # H2 = close + 1.1*(high-low)*1.1/6
    # H1 = close + 1.1*(high-low)*1.1/12
    # L1 = close - 1.1*(high-low)*1.1/12
    # L2 = close - 1.1*(high-low)*1.1/6
    # L3 = close - 1.1*(high-low)*1.1/4
    # L4 = close - 1.1*(high-low)*1.1/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot levels
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) * 1.1 / 4
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Load 1w data ONCE before loop for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        # If insufficient weekly data, use 1d EMA as fallback
        ema21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
        ema21_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d)
        weekly_uptrend = close > ema21_aligned
        weekly_downtrend = close < ema21_aligned
    else:
        # Pre-compute weekly EMA21 for trend filter
        close_1w = df_1w['close'].values
        ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
        ema21_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
        weekly_uptrend = close > ema21_aligned
        weekly_downtrend = close < ema21_aligned
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_sma_20_aligned[i]) or np.isnan(ema21_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 2x 20-period average
        vol_confirm = volume_current > 2.0 * volume_sma_20_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: price breaks above H3 with volume confirmation and weekly uptrend
        if close[i] > camarilla_h3_aligned[i] and vol_confirm and weekly_uptrend[i]:
            enter_long = True
        
        # Short: price breaks below L3 with volume confirmation and weekly downtrend
        if close[i] < camarilla_l3_aligned[i] and vol_confirm and weekly_downtrend[i]:
            enter_short = True
        
        # Exit conditions: price returns to pivot level (mean reversion)
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns below H3 (mean reversion)
            exit_long = close[i] < camarilla_h3_aligned[i]
        elif position == -1:
            # Exit short if price returns above L3 (mean reversion)
            exit_short = close[i] > camarilla_l3_aligned[i]
        
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