#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume spike and 1w trend filter
# - Camarilla levels from 1d: R3, S3, R4, S4 (based on prior 1d OHLC)
# - Long: price breaks above R4 with volume > 2x 20-period average on 6h
# - Short: price breaks below S4 with volume > 2x 20-period average on 6h
# - Trend filter: 1w EMA(50) slope positive for long, negative for short
# - Exit: price returns to 1d close (pivot point) or opposite Camarilla level
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-30 trades/year (50-120 total over 4 years) to stay within fee drag limits
# - Camarilla pivots work well in ranging markets; breakouts at R4/S4 with volume and trend filter capture strong moves

name = "6h_1w_camarilla_breakout_volume_v1"
timeframe = "6h"
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
    
    # Load 1d data ONCE before loop for Camarilla pivots (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return signals
    
    # Load 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute 1w EMA(50) slope for trend direction
    ema_slope_1w = np.zeros_like(ema_50_1w_aligned)
    ema_slope_1w[1:] = ema_50_1w_aligned[1:] - ema_50_1w_aligned[:-1]
    
    # Pre-compute 6h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute ATR for stoploss
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(volume_sma_20[i]) or np.isnan(atr_14[i]) or 
            np.isnan(ema_slope_1w[i]) or i >= len(ema_slope_1w)):
            signals[i] = 0.0
            continue
        
        # Get prior 1d OHLC for Camarilla calculation (must be completed 1d bar)
        # We need the 1d bar that closed before the current 6h bar
        # Use align_htf_to_ltf to get the completed 1d values shifted to 6h timeframe
        if i < 100:  # Ensure we have enough history
            signals[i] = 0.0
            continue
            
        # Calculate Camarilla levels from prior 1d bar
        # We'll get the prior 1d bar's OHLC using the aligned arrays
        # But since we need to calculate pivots, we do it on the 1d dataframe
        # and align the levels
        
        # For simplicity, we'll calculate the Camarilla levels for each 1d bar
        # and then align them to the 6h timeframe
        # This needs to be done outside the loop for efficiency
        
        # We'll move the Camarilla calculation outside the loop
        pass  # Placeholder - we'll implement properly below
    
    # Move Camarilla calculation outside the loop for efficiency
    # Calculate prior 1d OHLC for each 1d bar
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Camarilla levels: based on prior day's range
    # R4 = c + (h-l) * 1.1/2
    # R3 = c + (h-l) * 1.1/4
    # S3 = c - (h-l) * 1.1/4
    # S4 = c - (h-l) * 1.1/2
    # Pivot = (h+l+c)/3
    rng_1d = h_1d - l_1d
    r4_1d = c_1d + rng_1d * 1.1 / 2
    r3_1d = c_1d + rng_1d * 1.1 / 4
    s3_1d = c_1d - rng_1d * 1.1 / 4
    s4_1d = c_1d - rng_1d * 1.1 / 2
    pivot_1d = (h_1d + l_1d + c_1d) / 3
    
    # Align Camarilla levels to 6h timeframe (completed 1d bar only)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Re-initialize signals and position
    signals = np.zeros(n)
    position = 0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(volume_sma_20[i]) or np.isnan(atr_14[i]) or 
            np.isnan(ema_slope_1w[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(pivot_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Camarilla levels from prior completed 1d bar
        r4 = r4_aligned[i]
        s4 = s4_aligned[i]
        pivot = pivot_aligned[i]
        
        # Volume confirmation: current volume > 2x 20-period average
        vol_confirm = volume_current > 2.0 * volume_sma_20[i]
        
        # Trend filter: 1w EMA(50) slope
        ema_slope = ema_slope_1w[i]
        uptrend = ema_slope > 0
        downtrend = ema_slope < 0
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price above R4 with volume confirmation and uptrend
        if close_price > r4 and vol_confirm and uptrend:
            enter_long = True
        
        # Short breakout: price below S4 with volume confirmation and downtrend
        if close_price < s4 and vol_confirm and downtrend:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to pivot or ATR-based stop
            exit_long = (close_price <= pivot) or (close_price <= entry_price - 2.0 * atr_14[i])
        elif position == -1:
            # Exit short if price returns to pivot or ATR-based stop
            exit_short = (close_price >= pivot) or (close_price >= entry_price + 2.0 * atr_14[i])
        
        # Track entry price for stoploss calculation
        if enter_long or enter_short:
            entry_price = close_price
        
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