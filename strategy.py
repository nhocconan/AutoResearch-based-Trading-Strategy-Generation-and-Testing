#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with 1w volume confirmation and ADX(14) regime filter
# - Long when price breaks above H3 pivot level + 1w volume > 1.5x 20-period volume SMA + ADX < 25 (low volatility regime)
# - Short when price breaks below L3 pivot level + 1w volume > 1.5x 20-period volume SMA + ADX < 25
# - Exit: price returns to the daily pivot point (mean reversion to equilibrium)
# - Position sizing: 0.25 discrete level
# - Camarilla pivots identify key support/resistance levels, volume confirms breakout validity, ADX filter avoids choppy/strong trends
# - Works in bull/bear: breakouts effective in trending markets, ADX filter prevents false signals during low momentum periods

name = "1d_1w_camarilla_pivot_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate daily Camarilla pivot levels
    pivot = (high + low + close) / 3.0
    range_hl = high - low
    h4 = close + range_hl * 1.1 / 2.0
    h3 = close + range_hl * 1.1 / 4.0
    h2 = close + range_hl * 1.1 / 6.0
    h1 = close + range_hl * 1.1 / 12.0
    l1 = close - range_hl * 1.1 / 12.0
    l2 = close - range_hl * 1.1 / 6.0
    l3 = close - range_hl * 1.1 / 4.0
    l4 = close - range_hl * 1.1 / 2.0
    
    # Calculate 1d ADX(14) for regime filter
    # True Range
    tr1 = np.maximum(high - low, 
                     np.maximum(np.abs(high - np.roll(close, 1)), 
                                np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]
    # Plus Directional Movement
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    plus_dm[0] = 0
    # Minus Directional Movement
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    minus_dm[0] = 0
    # Smoothed values
    atr = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    # Handle division by zero and invalid values
    plus_di = np.where(atr == 0, 0, plus_di)
    minus_di = np.where(atr == 0, 0, minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = np.where(np.isnan(adx) | np.isinf(adx), 0, adx)
    
    # Calculate 1w volume SMA(20) for confirmation
    volume_1w = df_1w['volume'].values
    volume_sma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_sma_20_1w)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx[i]) or np.isnan(volume_sma_20_1w_aligned[i]) or 
            np.isnan(h3[i]) or np.isnan(l3[i]) or np.isnan(pivot[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1w volume > 1.5x 20-period SMA (volume spike)
        vol_1w_current = align_htf_to_ltf(prices, df_1w, df_1w['volume'].values)
        vol_confirm = vol_1w_current[i] > 1.5 * volume_sma_20_1w_aligned[i]
        
        # Regime filter: ADX < 25 indicates low volatility/non-trending market (favorable for breakout follow-through)
        low_volatility = adx[i] < 25
        
        # Breakout conditions
        breakout_long = close[i] > h3[i]  # Price breaks above H3 resistance
        breakout_short = close[i] < l3[i]  # Price breaks below L3 support
        
        # Exit conditions: price returns to daily pivot point (mean reversion)
        exit_long = close[i] < pivot[i]  # Exit long when price falls below pivot
        exit_short = close[i] > pivot[i]  # Exit short when price rises above pivot
        
        # Entry conditions: breakout with volume and regime confirmation
        long_entry = breakout_long and vol_confirm and low_volatility
        short_entry = breakout_short and vol_confirm and low_volatility
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals