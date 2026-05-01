#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend direction with 1w EMA34 filter and RSI(14) momentum confirmation
# KAMA adapts to market noise, reducing whipsaws in choppy markets. 1w EMA34 ensures alignment with weekly trend.
# RSI(14) > 50 for long, < 50 for short adds momentum confirmation. Discrete sizing 0.25 minimizes fee churn.
# Target: 30-100 total trades over 4 years (7-25/year). Works in bull (trend continuation) and bear (trend respect).

name = "1d_KAMA_Trend_1wEMA34_RSI_V1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for EMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # 1w EMA(34) on 1w close
    ema_1w_34 = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA to 1d timeframe
    ema_1w_34_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_34)
    
    # KAMA (Adaptive Moving Average) - ER = 10, fast = 2, slow = 30
    close_s = pd.Series(close)
    change = np.abs(close_s.diff(10))
    volatility = close_s.diff(1).abs().rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 30  # Need 30 for KAMA stability and RSI
    
    for i in range(start_idx, n):
        if np.isnan(ema_1w_34_aligned[i]) or np.isnan(kama[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Trend and momentum conditions
        kama_up = curr_close > kama[i]
        kama_down = curr_close < kama[i]
        rsi_long = rsi[i] > 50
        rsi_short = rsi[i] < 50
        weekly_trend_up = curr_close > ema_1w_34_aligned[i]
        weekly_trend_down = curr_close < ema_1w_34_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Price above KAMA, above 1w EMA34, RSI > 50
            if kama_up and weekly_trend_up and rsi_long:
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA, below 1w EMA34, RSI < 50
            elif kama_down and weekly_trend_down and rsi_short:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on price below KAMA or below 1w EMA34
            if curr_close < kama[i] or curr_close < ema_1w_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on price above KAMA or above 1w EMA34
            if curr_close > kama[i] or curr_close > ema_1w_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals