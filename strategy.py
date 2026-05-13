#1: Hypothesis
# Weekly pivot levels from 1w (major weekly support/resistance) act as strong institutional levels.
# On 1d timeframe: long when price breaks above weekly R3 with volume confirmation and price > weekly EMA20 (uptrend filter);
# short when price breaks below weekly S3 with volume confirmation and price < weekly EMA20 (downtrend filter).
# Exit when price returns to weekly pivot (PP) level or opposite extreme (S1/R1) is touched.
# Weekly levels reduce noise; daily breakouts capture momentum; volume confirms institutional interest.
# Designed for low trade frequency (<25/year) to avoid fee drag in ranging/bear markets (2025+ test).

#2: Implementation
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_Pivot_R3_S3_Breakout_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data (OHLC) once before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (standard formula)
    # PP = (H + L + C) / 3
    # R1 = 2*PP - L, S1 = 2*PP - H
    # R2 = PP + (H - L), S2 = PP - (H - L)
    # R3 = H + 2*(PP - L), S3 = L - 2*(H - PP)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    r2 = pp + (weekly_high - weekly_low)
    s2 = pp - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pp - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pp)
    
    # Align weekly levels to daily timeframe (wait for weekly bar to close)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Trend filter: weekly EMA20 (aligned to daily)
    weekly_ema20 = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema20)
    uptrend = close > ema20_aligned
    downtrend = close < ema20_aligned
    
    # Volume confirmation: > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: break above weekly R3 with volume and uptrend
            if close[i] > r3_aligned[i] and volume_confirm[i] and uptrend[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below weekly S3 with volume and downtrend
            elif close[i] < s3_aligned[i] and volume_confirm[i] and downtrend[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: return to weekly pivot or touch S1 (mean reversion)
            if close[i] <= pp_aligned[i] or close[i] <= s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: return to weekly pivot or touch R1 (mean reversion)
            if close[i] >= pp_aligned[i] or close[i] >= r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals