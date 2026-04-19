#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day KAMA direction combined with RSI momentum and weekly volatility regime filter.
# KAMA adapts to market noise - trends when efficiency ratio high, ranges when low.
# Long when: KAMA rising, RSI > 50, weekly ATR ratio < 1.0 (low volatility regime)
# Short when: KAMA falling, RSI < 50, weekly ATR ratio < 1.0
# Uses weekly ATR ratio to filter trades to low volatility periods where mean reversion works better.
# Weekly ATR ratio = current ATR(14) / average ATR(14) over past 52 weeks.
# Stable, low-trade-frequency strategy targeting 10-20 trades/year.
name = "1d_KAMA_RSI_VolatilityFilter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate KAMA (adaptive moving average)
    # Efficiency ratio = abs(net change) / sum of absolute changes
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = pd.Series(change).rolling(window=10, min_periods=10).sum().values
    net_change = np.abs(np.subtract(close, np.roll(close, 10)))
    net_change[:10] = 0  # first 10 values invalid
    er = np.divide(net_change, volatility, out=np.zeros_like(volatility), where=volatility!=0)
    # Smoothing constants
    sc = np.power(er * (0.66 - 0.06) + 0.06, 2)
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_loss), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Load weekly data for volatility regime filter (ATR ratio)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 60:
        return np.zeros(n)
    
    # Calculate weekly ATR(14)
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    tr1 = high_w - low_w
    tr2 = np.abs(high_w - np.roll(close_w, 1))
    tr3 = np.abs(low_w - np.roll(close_w, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_w = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate average ATR over past 52 weeks (1 year)
    atr_ma_52 = pd.Series(atr_w).rolling(window=52, min_periods=52).mean().values
    # Weekly ATR ratio (current vs annual average)
    atr_ratio_w = np.divide(atr_w, atr_ma_52, out=np.ones_like(atr_ma_52), where=atr_ma_52!=0)
    
    # Align weekly ATR ratio to daily timeframe (with 1-week delay for completed bar)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_weekly, atr_ratio_w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 52)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(atr_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Low volatility regime filter: weekly ATR ratio < 1.0 (below average volatility)
        low_vol_regime = atr_ratio_aligned[i] < 1.0
        
        if position == 0:
            # Long entry: KAMA rising, RSI > 50, low volatility regime
            if (kama[i] > kama[i-1] and rsi[i] > 50 and low_vol_regime):
                signals[i] = 0.25
                position = 1
            # Short entry: KAMA falling, RSI < 50, low volatility regime
            elif (kama[i] < kama[i-1] and rsi[i] < 50 and low_vol_regime):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA falling OR RSI < 40
            if (kama[i] < kama[i-1] or rsi[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA rising OR RSI > 60
            if (kama[i] > kama[i-1] or rsi[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals