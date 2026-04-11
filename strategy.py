#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return signals
    
    # Calculate weekly ATR for volatility filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range calculation
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(10) - using Wilder's smoothing (equivalent to RMA)
    atr_1w = np.zeros_like(tr)
    atr_1w[0] = tr[0]
    for i in range(1, len(tr)):
        atr_1w[i] = (atr_1w[i-1] * 9 + tr[i]) / 10
    
    # Align weekly ATR to daily timeframe
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Load daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Calculate Camarilla pivot levels from daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    daily_range = high_1d - low_1d
    
    # Key levels: R4 (resistance) and S4 (support)
    r4 = close_1d + (daily_range * 1.1 / 2)
    s4 = close_1d - (daily_range * 1.1 / 2)
    
    # Exit levels: R3 and S3
    r3 = close_1d + (daily_range * 1.1 / 4)
    s3 = close_1d - (daily_1d * 1.1 / 4)
    
    # Align daily levels to daily timeframe (no shift needed as already aligned)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: daily volume > 1.5x 20-day average (moderate threshold)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volatility filter: only trade when weekly ATR is above its 50-period average
        atr_ma_50 = pd.Series(atr_1w_aligned).rolling(window=50, min_periods=50).mean().values
        vol_filter = atr_1w_aligned[i] > atr_ma_50[i]
        
        # Volume confirmation - moderate threshold
        vol_confirm = volume_current > 1.5 * vol_ma_20[i]
        
        # Breakout conditions using Camarilla levels
        breakout_up = price_close > r4_aligned[i]  # Break above R4
        breakout_down = price_close < s4_aligned[i]  # Break below S4
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Break above R4 with volume confirmation and volatility filter
        if breakout_up and vol_confirm and vol_filter:
            enter_long = True
        
        # Short: Break below S4 with volume confirmation and volatility filter
        if breakout_down and vol_confirm and vol_filter:
            enter_short = True
        
        # Exit conditions: return to opposite S3/R3 levels
        exit_long = price_close < s3_aligned[i]  # Return to S3 level
        exit_short = price_close > r3_aligned[i]  # Return to R3 level
        
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

# Hypothesis: Daily Camarilla breakout strategy with weekly volatility filter and volume confirmation.
# Enters long when price breaks above weekly ATR-adjusted R4 with volume > 1.5x 20-day average.
# Enters short when price breaks below weekly ATR-adjusted S4 with volume > 1.5x 20-day average.
# Exits when price returns to S3/R3 levels respectively.
# Uses weekly ATR filter to avoid choppy markets and focus on volatile breakout periods.
# Position size set to 0.25 to balance risk and reward in volatile crypto markets.
# Target: 10-20 trades per year (40-80 total over 4 years) to minimize fee drag.
# Weekly timeframe for volatility filter ensures we only trade during high-volatility regimes.
# Works in both bull and bear markets by capturing significant breakouts in either direction.