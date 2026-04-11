#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_keltner_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # Calculate weekly Keltner Channel components
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly EMA(20) as the middle line
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Weekly ATR(10) for channel width
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr10_1w = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Upper and lower bands
    upper_1w = ema20_1w + 2 * atr10_1w
    lower_1w = ema20_1w - 2 * atr10_1w
    
    # Align to daily timeframe
    ema20_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper_1w)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower_1w)
    
    # Daily volume confirmation: volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema20_aligned[i]) or np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_current > 1.5 * vol_ma_20[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Price breaks above upper Keltner band + volume confirmation
        if price_close > upper_aligned[i] and vol_confirm:
            enter_long = True
        
        # Short: Price breaks below lower Keltner band + volume confirmation
        if price_close < lower_aligned[i] and vol_confirm:
            enter_short = True
        
        # Exit conditions: price returns to middle EMA
        exit_long = price_close < ema20_aligned[i]
        exit_short = price_close > ema20_aligned[i]
        
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

# Hypothesis: Keltner Breakout on weekly timeframe with daily execution. 
# Uses weekly EMA(20) as trend and ATR(10) for dynamic channel width.
# Breakouts above upper band signal strong uptrend, below lower band signal strong downtrend.
# Volume confirmation ensures breakouts have participation.
# Returns to middle EMA act as exit signals, capturing trends while avoiding whipsaws.
# Position size 0.25 limits drawdown. Target: 20-50 trades over 4 years (5-12/year).
# Works in both bull and break markets by capturing strong directional moves.