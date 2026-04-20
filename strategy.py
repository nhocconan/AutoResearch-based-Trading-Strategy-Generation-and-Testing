# 1h_Structure_And_Momentum
# Hypothesis: Combine market structure (HH/HL/LH/LL) with RSI momentum on 1h timeframe, using 4h trend filter and session filter (08-20 UTC).
# Market structure identifies swing points; RSI confirms momentum in the direction of structure.
# 4h EMA filter ensures trades align with higher timeframe trend.
# Session filter reduces noise during low-volume hours.
# Target: 15-35 trades/year per symbol to stay within fee limits.

name = "1h_Structure_And_Momentum"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # RSI(14) on 1h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Market structure: swing highs/lows (3-bar lookback)
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    for i in range(2, n-2):
        if high[i] > high[i-1] and high[i] > high[i-2] and high[i] > high[i+1] and high[i] > high[i+2]:
            swing_high[i] = True
        if low[i] < low[i-1] and low[i] < low[i-2] and low[i] < low[i+1] and low[i] < low[i+2]:
            swing_low[i] = True
    
    # Structure states: track last swing high/low
    last_swing_high = np.full(n, np.nan)
    last_swing_low = np.full(n, np.nan)
    last_high_idx = -1
    last_low_idx = -1
    
    for i in range(n):
        if swing_high[i]:
            last_high_idx = i
        if swing_low[i]:
            last_low_idx = i
        if last_high_idx != -1:
            last_swing_high[i] = high[last_high_idx]
        if last_low_idx != -1:
            last_swing_low[i] = low[last_low_idx]
    
    # Structure signals: bullish if price above last swing low, bearish if below last swing high
    bullish_structure = close > last_swing_low
    bearish_structure = close < last_swing_high
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure RSI and structure are ready
    
    for i in range(start_idx, n):
        # Skip if required data is not ready
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(last_swing_high[i]) or np.isnan(last_swing_low[i])):
            signals[i] = 0.0
            continue
        
        # Require session
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: bullish structure + RSI > 50 + price above 4h EMA (uptrend)
            if bullish_structure[i] and rsi[i] > 50 and close[i] > ema_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: bearish structure + RSI < 50 + price below 4h EMA (downtrend)
            elif bearish_structure[i] and rsi[i] < 50 and close[i] < ema_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long: exit if structure turns bearish or RSI < 40
            if not bullish_structure[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short: exit if structure turns bullish or RSI > 60
            if not bearish_structure[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals