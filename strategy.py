# 4h_12h_camarilla_breakout_volatility_filter
# Uses 12h Camarilla pivot levels (H4/L4) as support/resistance on 4h chart.
# Long when price breaks above H4 with low volatility (ATR < 1.5x 20-period ATR).
# Short when price breaks below L4 with low volatility.
# Exits when price returns to 12h pivot point (mean reversion).
# Designed for low trade frequency (target: 20-30 per year) to minimize fee drag.
# Works in trending markets via breakouts and in ranging markets via mean reversion.
# Filters out high-volatility breakouts that often fail, improving win rate.

from __future__ import annotations
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_breakout_volatility_filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels (based on previous 12h bar's OHLC)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate pivot point and Camarilla levels for each 12h period
    pp = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    
    # Camarilla levels: H4 = PP + 1.1/2 * range, L4 = PP - 1.1/2 * range
    h4 = pp + (1.1 / 2) * range_12h
    l4 = pp - (1.1 / 2) * range_12h
    
    # Align 12h levels to 4h timeframe (12h values update after 12h bar closes)
    h4_aligned = align_htf_to_ltf(prices, df_12h, h4)
    l4_aligned = align_htf_to_ltf(prices, df_12h, l4)
    pp_aligned = align_htf_to_ltf(prices, df_12h, pp)
    
    # Volatility filter: ATR < 1.5 * 20-period ATR (4h timeframe)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    vol_filter = atr < (1.5 * pd.Series(atr).rolling(window=20, min_periods=20).mean().values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or np.isnan(pp_aligned[i]) or np.isnan(vol_filter[i]):
            signals[i] = 0.0
            continue
        
        # Require low volatility for new entries
        if not vol_filter[i]:
            # Hold current position if volatility filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above H4 with low volatility
        if close[i] > h4_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below L4 with low volatility
        elif close[i] < l4_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: price returns to 12h pivot point (mean reversion)
        elif position == 1 and close[i] <= pp_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= pp_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals