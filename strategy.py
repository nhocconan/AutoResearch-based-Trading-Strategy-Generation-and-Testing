#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Supertrend for trend direction and 1h RSI(2) for precise entry timing
# Long when 4h Supertrend is bullish AND 1h RSI(2) crosses below 10 (extreme oversold)
# Short when 4h Supertrend is bearish AND 1h RSI(2) crosses above 90 (extreme overbought)
# Exit when RSI(2) crosses back through 50 (mean reversion) or Supertrend flips
# Uses discrete sizing 0.20 to balance return and risk
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# 4h Supertrend provides reliable trend filter to avoid counter-trend trades
# 1h RSI(2) enables timely entries on pullbacks in trending markets
# Session filter (08-20 UTC) reduces noise and improves win rate
# Works in bull markets (buying oversold pullbacks in uptrend) and bear markets (selling overbought bounces in downtrend)

name = "1h_Supertrend4h_RSI2_Entry"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for Supertrend calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:  # Need enough for ATR calculation
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ATR(10) for Supertrend
    tr1 = pd.Series(high_4h).rolling(2).max() - pd.Series(low_4h).rolling(2).min()
    tr2 = abs(pd.Series(high_4h).rolling(2).apply(lambda x: x[0] - x[1] if len(x)==2 else 0, raw=True))
    tr3 = abs(pd.Series(low_4h).rolling(2).apply(lambda x: x[0] - x[1] if len(x)==2 else 0, raw=True))
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).fillna(0).values
    atr_4h = pd.Series(tr_4h).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate Supertrend
    hl2_4h = (high_4h + low_4h) / 2
    upper_4h = hl2_4h + (3 * atr_4h)
    lower_4h = hl2_4h - (3 * atr_4h)
    
    upper_4h = np.where(pd.Series(upper_4h).rolling(2, min_periods=1).min() < upper_4h, 
                        pd.Series(upper_4h).rolling(2, min_periods=1).min(), upper_4h)
    lower_4h = np.where(pd.Series(lower_4h).rolling(2, min_periods=1).max() > lower_4h, 
                        pd.Series(lower_4h).rolling(2, min_periods=1).max(), lower_4h)
    
    supertrend_4h = np.full_like(close_4h, np.nan, dtype=float)
    supertrend_4h[0] = upper_4h[0]
    for i in range(1, len(close_4h)):
        if close_4h[i-1] <= supertrend_4h[i-1]:
            supertrend_4h[i] = min(upper_4h[i], supertrend_4h[i-1])
        else:
            supertrend_4h[i] = max(lower_4h[i], supertrend_4h[i-1])
    
    # Determine trend direction: 1 for bullish (price > supertrend), -1 for bearish (price < supertrend)
    trend_4h = np.where(close_4h > supertrend_4h, 1, -1)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Calculate RSI(2) on 1h timeframe
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # Neutral when undefined
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(2, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(trend_4h_aligned[i]) or np.isnan(rsi[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 4h Supertrend bullish AND 1h RSI(2) crosses below 10 (extreme oversold)
            if (trend_4h_aligned[i] == 1 and 
                rsi[i] < 10 and rsi[i-1] >= 10):
                signals[i] = 0.20
                position = 1
            # Short: 4h Supertrend bearish AND 1h RSI(2) crosses above 90 (extreme overbought)
            elif (trend_4h_aligned[i] == -1 and 
                  rsi[i] > 90 and rsi[i-1] <= 90):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: RSI(2) crosses back above 50 OR Supertrend turns bearish
            if rsi[i] > 50 or trend_4h_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: RSI(2) crosses back below 50 OR Supertrend turns bullish
            if rsi[i] < 50 or trend_4h_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals