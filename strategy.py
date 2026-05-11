# 1d_1W_RSI_Momentum_With_WeeklyTrend
# Hypothesis: On daily timeframe, use RSI momentum (RSI > 60 for long, < 40 for short) filtered by weekly trend (price above/below weekly EMA50) and volume confirmation. This captures momentum moves in both bull and bear markets while avoiding counter-trend trades. Weekly trend filter reduces whipsaws, volume surge ensures institutional participation. Targeting 15-25 trades/year to minimize fee drag on 1d timeframe.
#!/usr/bin/env python3
name = "1d_1W_RSI_Momentum_With_WeeklyTrend"
timeframe = "1d"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume ratio (20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold
        volume_surge = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: RSI > 60 (momentum), price above weekly EMA50 (uptrend), volume surge
            if (rsi[i] > 60 and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_surge):
                signals[i] = 0.25
                position = 1
            # Short: RSI < 40 (momentum), price below weekly EMA50 (downtrend), volume surge
            elif (rsi[i] < 40 and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_surge):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: RSI < 50 (momentum fade) or trend turns bearish
                if (rsi[i] < 50) or (close[i] < ema_50_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: RSI > 50 (momentum fade) or trend turns bullish
                if (rsi[i] > 50) or (close[i] > ema_50_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals