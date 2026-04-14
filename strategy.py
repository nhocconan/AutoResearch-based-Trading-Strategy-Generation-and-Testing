# ANALYSIS & STRATEGY DESIGN
# Objective: Create a robust 1h strategy with minimal trade frequency (target: 15-37 trades/year) to avoid fee drag.
# Core Idea: Use 4h trend direction (EMA crossover) as the primary signal filter, and 1h for precise entry timing using
# price action relative to the 4h EMA. This ensures trades only occur in the direction of the higher timeframe trend,
# reducing whipsaws. Volume confirmation adds robustness.
# Risk Management: Fixed position size (0.20) and trend-following exits (when 4h EMA slope changes sign).
# Session Filter: Limit trading to 08:00-20:00 UTC to avoid low-liquidity periods.
# Expected Performance: Low trade count, high win rate in trending markets, resilience in ranging/choppy markets due to
# strict 4h trend filter.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE for trend and filters
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # 4h EMA(21) for trend direction
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    # 4h EMA slope: current EMA - previous EMA
    ema_slope_4h = np.diff(ema_4h, prepend=np.nan)
    
    # 4h ATR(14) for volatility filter
    tr1 = np.abs(high_4h[1:] - low_4h[1:])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_4h = np.concatenate([[np.nan], tr_4h])
    atr_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 4h EMA(50) for dynamic support/resistance
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h indicators to 1h timeframe
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    ema_slope_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_slope_4h)
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1h EMA(20) for entry timing
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 1h volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20  # Fixed 20% position size
    
    # Precompute session filter (08:00-20:00 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Start after sufficient warmup
    start = max(50, 20)
    
    for i in range(start, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any critical data is NaN
        if (np.isnan(ema_4h_aligned[i]) or 
            np.isnan(ema_slope_4h_aligned[i]) or
            np.isnan(atr_4h_aligned[i]) or
            np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(ema_20[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        # 4h trend filter: only trade in direction of 4h EMA slope
        uptrend_4h = ema_slope_4h_aligned[i] > 0
        downtrend_4h = ema_slope_4h_aligned[i] < 0
        
        if position == 0:
            # Look for entry opportunities
            # Long: price above 4h EMA(21) and 4h EMA(50), with volume and uptrend
            if (close[i] > ema_4h_aligned[i] and 
                close[i] > ema_50_4h_aligned[i] and
                uptrend_4h and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price below 4h EMA(21) and 4h EMA(50), with volume and downtrend
            elif (close[i] < ema_4h_aligned[i] and 
                  close[i] < ema_50_4h_aligned[i] and
                  downtrend_4h and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below 4h EMA(21) or 4h trend turns down
            if (close[i] < ema_4h_aligned[i] or 
                ema_slope_4h_aligned[i] <= 0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above 4h EMA(21) or 4h trend turns up
            if (close[i] > ema_4h_aligned[i] or 
                ema_slope_4h_aligned[i] >= 0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4hEMA_Trend_Filter_Volume_Entry"
timeframe = "1h"
leverage = 1.0