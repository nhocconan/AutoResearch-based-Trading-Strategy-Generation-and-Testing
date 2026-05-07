#!/usr/bin/env python3
name = "1d_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 250:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Daily Camarilla levels
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    calc_range = prev_high - prev_low
    S3 = prev_close - calc_range * 1.1666
    S2 = prev_close - calc_range * 1.0833
    S1 = prev_close - calc_range * 1.0
    R1 = prev_close + calc_range * 1.0
    R2 = prev_close + calc_range * 1.0833
    R3 = prev_close + calc_range * 1.1666
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 250
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(S3[i]) or np.isnan(R3[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R3 with volume in weekly uptrend
            if close[i] > R3[i] and volume[i] > vol_ma_20[i] * 2.0 and ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]:
                signals[i] = 0.30
                position = 1
            # Short: break below S3 with volume in weekly downtrend
            elif close[i] < S3[i] and volume[i] > vol_ma_20[i] * 2.0 and ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: close below R1 or weekly trend reverses
            if close[i] < R1[i] or ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: close above S1 or weekly trend reverses
            if close[i] > S1[i] or ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: 1d Camarilla R3/S3 breakouts with weekly EMA34 trend filter and volume confirmation
# - Camarilla levels (R3/S3) act as strong support/resistance derived from prior day's range
# - Breakout above R3 or below S3 with 2x average volume signals institutional participation
# - Weekly EMA34 trend filter ensures alignment with higher timeframe momentum
# - Exits on retracement to R1/S1 or weekly trend reversal to avoid giving back profits
# - Works in bull markets (long breakouts in uptrend) and bear markets (short breakdowns in downtrend)
# - Position size 0.30 balances profit potential with risk control
# - Target: 20-50 trades/year to stay within fee drag limits
# - Proven pattern: Camarilla + volume + trend showed strong performance in DB (up to 2.055 Sharpe)