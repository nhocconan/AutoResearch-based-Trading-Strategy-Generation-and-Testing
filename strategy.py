#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 12h volume confirmation and chop regime filter
# - Primary: 4h price breaking above/below Camarilla H3/L3 levels from prior 12h session
# - HTF: 12h volume confirmation (current volume > 1.3x 20-period MA) + chop regime (CHOP > 50 for ranging bias)
# - Long: Breakout above H3 + volume confirmation + chop > 50 (favor mean reversion in chop)
# - Short: Breakout below L3 + volume confirmation + chop > 50
# - Exit: Price returns to H4/L4 levels (opposite side of pivot structure)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Camarilla captures intraday reversals, volume confirms conviction, chop filter avoids strong trends
# - Target: 25-40 trades/year over 4 years (100-160 total) to stay within fee drag limits

name = "4h_12h_camarilla_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough data for indicators
        return np.zeros(n)
    
    # Pre-compute 4h data
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    
    # Pre-compute 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h Camarilla pivot levels (based on prior 12h bar)
    # Camarilla: H4 = close + 1.1*(high-low)/2, H3 = close + 1.1*(high-low)/4
    #          L3 = close - 1.1*(high-low)/4, L4 = close - 1.1*(high-low)/2
    hl_range = high_12h - low_12h
    camarilla_h4 = close_12h + 1.1 * hl_range / 2.0
    camarilla_h3 = close_12h + 1.1 * hl_range / 4.0
    camarilla_l3 = close_12h - 1.1 * hl_range / 4.0
    camarilla_l4 = close_12h - 1.1 * hl_range / 2.0
    
    # Calculate 12h Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(N)
    period = 14
    tr1 = np.abs(np.roll(high_12h, 1) - np.roll(low_12h, 1))
    tr2 = np.abs(np.roll(high_12h, 1) - np.roll(close_12h, 1))
    tr3 = np.abs(np.roll(low_12h, 1) - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    max_high = pd.Series(high_12h).rolling(window=period, min_periods=period).max().values
    min_low = pd.Series(low_12h).rolling(window=period, min_periods=period).min().values
    chop = 100 * np.log10(atr / (max_high - min_low)) / np.log10(period)
    
    # Calculate 12h volume moving average (20-period) for volume confirmation
    volume_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    volume_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_20_12h)
    volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period for all indicators
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(volume_ma_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.3x 20-period MA
        volume_confirm = volume_12h_aligned[i] > 1.3 * volume_ma_20_12h_aligned[i]
        
        # Chop regime filter: CHOP > 50 indicates ranging/choppy market (favor mean reversion)
        chop_confirm = chop_aligned[i] > 50.0
        
        # Camarilla breakout conditions (using prior 12h levels)
        breakout_long = close_4h[i] > camarilla_h3_aligned[i]
        breakout_short = close_4h[i] < camarilla_l3_aligned[i]
        
        # Exit conditions: Price returns to H4/L4 levels (opposite side)
        exit_long = close_4h[i] < camarilla_h4_aligned[i]
        exit_short = close_4h[i] > camarilla_l4_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Breakout above H3 + volume confirmation + chop confirmation
            if breakout_long and volume_confirm and chop_confirm:
                position = 1
                signals[i] = 0.25
            # Short entry: Breakout below L3 + volume confirmation + chop confirmation
            elif breakout_short and volume_confirm and chop_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price returns to opposite H4/L4 level
            if position == 1:  # Long position
                if exit_long:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if exit_short:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals