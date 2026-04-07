#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Williams %R with 1-week EMA trend filter
# Long when Williams %R < -80 (oversold) + price > weekly EMA50 (uptrend)
# Short when Williams %R > -20 (overbought) + price < weekly EMA50 (downtrend)
# Exit when Williams %R crosses -50 (mean reversion)
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses Williams %R for mean reversion entries and weekly EMA for trend filter
# Target: 30-100 total trades over 4 years (7-25/year)

name = "1d_williamsr_1w_ema_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1-week data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1-week EMA50
    close_1w = df_1w['close'].values
    close_1w_s = pd.Series(close_1w)
    ema50_1w = close_1w_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(14, n):
        # Skip if required data not available
        if (np.isnan(williams_r[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Williams %R crosses -50 (mean reversion)
            elif williams_r[i] >= -50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Williams %R crosses -50 (mean reversion)
            elif williams_r[i] <= -50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Williams %R extremes with trend filter
            # Long: Williams %R oversold + price above weekly EMA50 (uptrend)
            if williams_r[i] < -80 and close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Williams %R overbought + price below weekly EMA50 (downtrend)
            elif williams_r[i] > -20 and close[i] < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals