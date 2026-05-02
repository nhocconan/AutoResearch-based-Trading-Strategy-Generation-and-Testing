#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation
# Uses 12h timeframe for trend direction and 6h for entry timing via Camarilla levels
# Volume confirmation (1.5x 20-period average) ensures institutional participation
# Target: 50-150 total trades over 4 years = 12-37/year for 6h timeframe
# Works in bull markets via trend-aligned breakouts, in bear via strict entry filters avoiding false signals
# Discrete position sizing (0.25) to minimize fee churn and control drawdown

name = "6h_Camarilla_R3S3_Breakout_12hEMA50_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla pivot levels from previous 6h bar
    # Typical Price = (H + L + C)/3
    typical_price = (high + low + close) / 3.0
    # Camarilla levels based on previous bar's range
    rng = high - low
    camarilla_h4 = typical_price + 1.1 * rng / 2.0  # R3
    camarilla_l4 = typical_price - 1.1 * rng / 2.0  # S3
    camarilla_h2 = typical_price + 1.1 * rng / 6.0  # R1
    camarilla_l2 = typical_price - 1.1 * rng / 6.0  # S1
    
    # Shift to align with bar close (use previous bar's levels)
    camarilla_h4 = np.roll(camarilla_h4, 1)
    camarilla_l4 = np.roll(camarilla_l4, 1)
    camarilla_h2 = np.roll(camarilla_h2, 1)
    camarilla_l2 = np.roll(camarilla_l2, 1)
    camarilla_h4[0] = np.nan
    camarilla_l4[0] = np.nan
    camarilla_h2[0] = np.nan
    camarilla_l2[0] = np.nan
    
    # Volume confirmation (1.5x 20-period average on 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or
            np.isnan(camarilla_h2[i]) or np.isnan(camarilla_l2[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Camarilla H4 (R3) + price > 12h EMA50 + volume confirm
            if close[i] > camarilla_h4[i] and close[i] > ema_50_12h_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla L4 (S3) + price < 12h EMA50 + volume confirm
            elif close[i] < camarilla_l4[i] and close[i] < ema_50_12h_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Camarilla H2 (R1) or trend reversal
            if close[i] < camarilla_h2[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Camarilla L2 (S1) or trend reversal
            if close[i] > camarilla_l2[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals