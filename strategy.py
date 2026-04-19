#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Supertrend with 1-week ATR-based trend filter and volume confirmation.
# Long when: Supertrend flips below price (green), 1w ATR > 1.5x 10-period average (high volatility regime), volume > 1.2x 20-period average
# Short when: Supertrend flips above price (red), 1w ATR > 1.5x 10-period average, volume > 1.2x 20-period average
# Exit when Supertrend flips to opposite side.
# Supertrend adapts to volatility, capturing trends while avoiding whipsaws in low-volatility regimes.
# High volatility filter ensures we only trade when trends are strong enough to overcome noise.
# Target: 20-40 trades/year per symbol.
name = "6h_Supertrend_Vol_VolatilityFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-week data for volatility filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ATR on weekly data
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1w = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr_1w_ma = pd.Series(atr_1w).rolling(window=10, min_periods=10).mean().values
    
    # Align ATR and its MA to 6h timeframe
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    atr_1w_ma_aligned = align_htf_to_ltf(prices, df_1w, atr_1w_ma)
    
    # Calculate Supertrend on 6h data
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Basic Upper and Lower Bands
    basic_ub = (high + low) / 2 + multiplier * atr
    basic_lb = (high + low) / 2 - multiplier * atr
    
    # Final Upper and Lower Bands
    final_ub = np.full(n, np.nan)
    final_lb = np.full(n, np.nan)
    for i in range(n):
        if i == 0:
            final_ub[i] = basic_ub[i]
            final_lb[i] = basic_lb[i]
        else:
            if close[i-1] <= final_ub[i-1]:
                final_ub[i] = min(basic_ub[i], final_ub[i-1])
            else:
                final_ub[i] = basic_ub[i]
            if close[i-1] >= final_lb[i-1]:
                final_lb[i] = max(basic_lb[i], final_lb[i-1])
            else:
                final_lb[i] = basic_lb[i]
    
    # Supertrend and Direction
    supertrend = np.full(n, np.nan)
    trend = np.zeros(n)  # 1 for uptrend, -1 for downtrend
    for i in range(n):
        if i == 0:
            supertrend[i] = final_ub[i]
            trend[i] = 1
        else:
            if trend[i-1] == 1 and close[i] <= final_ub[i]:
                trend[i] = -1
                supertrend[i] = final_lb[i]
            elif trend[i-1] == -1 and close[i] >= final_lb[i]:
                trend[i] = 1
                supertrend[i] = final_ub[i]
            else:
                trend[i] = trend[i-1]
                supertrend[i] = final_ub[i] if trend[i] == 1 else final_lb[i]
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(supertrend[i]) or np.isnan(trend[i]) or 
            np.isnan(atr_1w_aligned[i]) or np.isnan(atr_1w_ma_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        st = supertrend[i]
        tr = trend[i]
        vol_atr = atr_1w_aligned[i]
        vol_atr_ma = atr_1w_ma_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: Supertrend green (below price), high volatility regime, volume confirmation
            if price > st and tr == 1 and vol_atr > 1.5 * vol_atr_ma and vol > 1.2 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short entry: Supertrend red (above price), high volatility regime, volume confirmation
            elif price < st and tr == -1 and vol_atr > 1.5 * vol_atr_ma and vol > 1.2 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Supertrend flips red (above price)
            if price < st:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Supertrend flips green (below price)
            if price > st:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals