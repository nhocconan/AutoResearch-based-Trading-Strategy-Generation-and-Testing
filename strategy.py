#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d_1w_camarilla_breakout_v2
# Uses weekly Camarilla pivot levels (H4/L4) as key support/resistance on daily chart.
# Long when price breaks above H4 with volume confirmation (volume > 1.5x 20-day avg).
# Short when price breaks below L4 with volume confirmation.
# Exits when price returns to weekly pivot point (PP).
# Added volatility filter: only trade when daily ATR(10) > 1.2x 50-day average to avoid chop.
# Designed for low trade frequency (target: 10-20 trades/year) to minimize fee drift.
# Works in trending markets via breakouts and in ranging markets via mean reversion to pivot.

name = "1d_1w_camarilla_breakout_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels based on previous week's OHLC
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot point and Camarilla levels for each week
    pp = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    # Camarilla levels: H4 = PP + 1.1/2 * range, L4 = PP - 1.1/2 * range
    h4 = pp + (1.1 / 2) * range_1w
    l4 = pp - (1.1 / 2) * range_1w
    
    # Align weekly levels to daily timeframe (weekly values update after weekly bar closes)
    h4_aligned = align_htf_to_ltf(prices, df_1w, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1w, l4)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    
    # Volume confirmation: volume > 1.5 * 20-period average (daily timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    # Volatility filter: ATR(10) > 1.2 * 50-day average ATR to avoid chop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR
    atr10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    vol_filter = atr10 > (atr50 * 1.2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or np.isnan(pp_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Require both volume and volatility filters for new entries
        if not (vol_confirm[i] and vol_filter[i]):
            # Hold current position if filters fail
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above H4
        if close[i] > h4_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below L4
        elif close[i] < l4_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: price returns to weekly pivot point (mean reversion)
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