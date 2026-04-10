#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA crossover with 4h trend filter + volume spike + session filter
# - Primary signal: 1h EMA(9) crosses above/below EMA(21) for entry timing
# - Trend filter: 4h EMA(50) slope (must be aligned to 1h) - only trade in trend direction
# - Volume confirmation: 1h volume > 1.5x 20-period average volume (avoid low-participation signals)
# - Session filter: Trade only during 08:00-20:00 UTC to avoid low-liquidity hours
# - Works in bull/bear: Trend filter adapts to market regime; volume confirms participation
# - Position size: 0.20 discrete level to minimize fee churn
# - Target: 15-30 trades/year (60-120 total over 4 years) per 1h strategy guidelines
# - ATR-based stoploss: exit when price moves against position by 2.0x ATR(14)

name = "1h_4h_ema_cross_volume_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    # Pre-compute 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Pre-compute 4h EMA(50) previous value for slope
    ema_50_4h_prev = np.roll(ema_50_4h_aligned, 1)
    ema_50_4h_prev[0] = ema_50_4h_aligned[0]
    ema_50_slope = ema_50_4h_aligned - ema_50_4h_prev  # positive = uptrend, negative = downtrend
    
    # Pre-compute 1h EMA(9) and EMA(21) for crossover
    close_1h = prices['close'].values
    ema_9 = pd.Series(close_1h).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_21 = pd.Series(close_1h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Pre-compute 1h volume spike filter
    volume_1h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1h > (1.5 * avg_volume_20)
    
    # Pre-compute 1h ATR(14) for stoploss
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    tr1 = high_1h - low_1h
    tr2 = np.abs(high_1h - np.roll(close_1h, 1))
    tr3 = np.abs(low_1h - np.roll(close_1h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute session filter (08:00-20:00 UTC)
    # open_time is already datetime64[ns], access hour via index
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_9[i]) or np.isnan(ema_21[i]) or
            np.isnan(ema_50_slope[i]) or np.isnan(volume_spike[i]) or
            np.isnan(atr_14[i]) or np.isnan(in_session[i])):
            signals[i] = 0.0
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: EMA cross down OR stoploss hit
            if ema_9[i] < ema_21[i] or close_1h[i] < entry_price - 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: EMA cross up OR stoploss hit
            if ema_9[i] > ema_21[i] or close_1h[i] > entry_price + 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for EMA crossovers with trend filter, volume spike, and session
            # Only long in uptrend (EMA50 slope > 0), only short in downtrend (EMA50 slope < 0)
            if volume_spike[i]:
                # Bullish crossover: EMA9 crosses above EMA21
                if ema_9[i] > ema_21[i] and ema_9[i-1] <= ema_21[i-1]:
                    if ema_50_slope[i] > 0:  # only long in uptrend
                        position = 1
                        entry_price = close_1h[i]
                        signals[i] = 0.20
                # Bearish crossover: EMA9 crosses below EMA21
                elif ema_9[i] < ema_21[i] and ema_9[i-1] >= ema_21[i-1]:
                    if ema_50_slope[i] < 0:  # only short in downtrend
                        position = -1
                        entry_price = close_1h[i]
                        signals[i] = -0.20
    
    return signals