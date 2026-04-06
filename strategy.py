#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6s price action combined with weekly pivot structure and volume confirmation.
# Uses weekly pivot points as key support/resistance levels for mean reversion and breakout signals.
# Long when price rejects weekly S1/S2 with volume confirmation.
# Short when price rejects weekly R1/R2 with volume confirmation.
# Uses 6s EMA for trend filter to avoid counter-trend trades.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled risk.

name = "6s_weeklypivot_volume_meanrev_v1"
timeframe = "6s"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week
    # Standard pivot: P = (H + L + C) / 3
    # Support 1: S1 = (2*P) - H
    # Resistance 1: R1 = (2*P) - L
    # Support 2: S2 = P - (H - L)
    # Resistance 2: R2 = P + (H - L)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot calculations
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_hl_1w = high_1w - low_1w
    
    r1_1w = (2 * pivot_1w) - high_1w
    s1_1w = (2 * pivot_1w) - low_1w
    r2_1w = pivot_1w + range_hl_1w
    s2_1w = pivot_1w - range_hl_1w
    
    # Align to 6s timeframe (shifted by 1 week for prior week's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # 6s EMA for trend filter
    ema_fast = pd.Series(close).ewm(span=9, min_periods=9, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # Volume filters
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > vol_ma  # Volume above average
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(21, n):
        # Skip if required data not available
        if (np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR below entry
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above R1 or trend turns down
            elif close[i] > r1_aligned[i] or ema_fast[i] < ema_slow[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR above entry
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below S1 or trend turns up
            elif close[i] < s1_aligned[i] or ema_fast[i] > ema_slow[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for mean reversion entries at weekly support/resistance
            if vol_filter[i]:
                # Long setup: price near S1 or S2 with bullish rejection
                near_s1 = abs(close[i] - s1_aligned[i]) / s1_aligned[i] < 0.002  # Within 0.2%
                near_s2 = abs(close[i] - s2_aligned[i]) / s2_aligned[i] < 0.002
                
                # Short setup: price near R1 or R2 with bearish rejection
                near_r1 = abs(close[i] - r1_aligned[i]) / r1_aligned[i] < 0.002
                near_r2 = abs(close[i] - r2_aligned[i]) / r2_aligned[i] < 0.002
                
                # Rejection signals: price moving away from level after touching it
                if i > 1:
                    # Long rejection: was at/below support, now moving up
                    long_rejection = ((close[i-1] <= s1_aligned[i-1] * 1.002 or close[i-1] <= s2_aligned[i-1] * 1.002) and 
                                    close[i] > close[i-1])
                    # Short rejection: was at/above resistance, now moving down
                    short_rejection = ((close[i-1] >= r1_aligned[i-1] * 0.998 or close[i-1] >= r2_aligned[i-1] * 0.998) and 
                                     close[i] < close[i-1])
                    
                    if long_rejection and (near_s1 or near_s2):
                        signals[i] = 0.25
                        position = 1
                        entry_price = close[i]
                    elif short_rejection and (near_r1 or near_r2):
                        signals[i] = -0.25
                        position = -1
                        entry_price = close[i]
    
    return signals