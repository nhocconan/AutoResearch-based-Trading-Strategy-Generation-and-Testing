# 4D_SMART_MONEY_CONFIRMATION
# Hypothesis: Smart money leaves footprints via volume spikes at key structural levels (4h swing highs/lows).
# We detect accumulation/distribution by combining: 1) 4h swing points (fractals), 2) Volume spikes (>2x 20-period avg),
# 3) 1-day trend filter (price vs EMA50). Works in bull/bear by following institutional flow.
# Target: 25-40 trades/year (~100-160 total) to minimize fee drag while capturing high-probability moves.

name = "4D_SMART_MONEY_CONFIRMATION"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d trend filter: EMA(50) on close
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average (approx 10 hours)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Swing point detection (fractals) - 4-bar pattern: low-low-high-high for resistance, high-high-low-low for support
    # Bearish swing (resistance): high[-2] > high[-3] and high[-2] > high[-1] and high[-2] > high[-4]
    # Bullish swing (support): low[-2] < low[-3] and low[-2] < low[-1] and low[-2] < low[-4]
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    
    for i in range(4, n):
        # Bearish swing high (resistance level)
        if (high[i-2] > high[i-3] and 
            high[i-2] > high[i-1] and 
            high[i-2] > high[i-4]):
            swing_high[i-2] = True
        # Bullish swing low (support level)
        if (low[i-2] < low[i-3] and 
            low[i-2] < low[i-1] and 
            low[i-2] < low[i-4]):
            swing_low[i-2] = True
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(5, n):  # Start after warmup for swing detection
        if position == 0:
            # LONG: Price approaches swing low support with volume spike and uptrend
            if (swing_low[i] and 
                close[i] <= low[i] * 1.002 and  # Near swing low (within 0.2%)
                volume_spike[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price approaches swing high resistance with volume spike and downtrend
            elif (swing_high[i] and 
                  close[i] >= high[i] * 0.998 and  # Near swing high (within 0.2%)
                  volume_spike[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches swing high resistance or trend reverses
            if (swing_high[i] and close[i] >= high[i] * 0.998) or \
               (close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches swing low support or trend reverses
            if (swing_low[i] and close[i] <= low[i] * 1.002) or \
               (close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals