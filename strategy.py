# [EXPERIMENT #159094] 1H_CAMARILLA_R1_S1_BREAKOUT_4HTREND_1DVOLUME
# Hypothesis: 1h strategy using 4h trend filter and 1d volume confirmation to reduce false breakouts.
# Long: Price above 4h EMA50 + breaks above 1d R1 + volume spike (2x 20-period avg)
# Short: Price below 4h EMA50 + breaks below 1d S1 + volume spike
# Session filter: 08-20 UTC to avoid low-liquidity hours.
# Position size: 0.20 (20%) to manage drawdown.
# Target: 15-30 trades/year (60-120 total over 4 years) to stay within 1h limits.
# Works in bull markets (trend-following breaks) and bear markets (mean-reversion from extremes).
name = "1H_CAMARILLA_R1_S1_BREAKOUT_4HTREND_1DVOLUME"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    rang = prev_high - prev_low
    R1 = prev_close + rang * 1.1 / 4
    S1 = prev_close - rang * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup for EMA and volume MA
        if (np.isnan(ema_4h_aligned[i]) or 
            np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        in_session = (8 <= hours[i] <= 20)
        
        if position == 0:
            # LONG: Price above 4h EMA50 (bullish trend) + break above R1 + volume spike + session
            if (in_session and 
                close[i] > ema_4h_aligned[i] and 
                close[i] > R1_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price below 4h EMA50 (bearish trend) + break below S1 + volume spike + session
            elif (in_session and 
                  close[i] < ema_4h_aligned[i] and 
                  close[i] < S1_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters range OR closes below 4h EMA50
            if (close[i] < R1_aligned[i] and close[i] > S1_aligned[i]) or \
               close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price re-enters range OR closes above 4h EMA50
            if (close[i] < R1_aligned[i] and close[i] > S1_aligned[i]) or \
               close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals