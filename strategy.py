# 1d_WeeklyTrend_RSI_Pullback
# Hypothesis: On daily timeframe, buy pullbacks to EMA50 during weekly uptrends (price > weekly EMA200) and sell rallies to EMA50 during weekly downtrends (price < weekly EMA200).
# Uses RSI(14) for entry timing: RSI < 40 for longs in uptrend, RSI > 60 for shorts in downtrend.
# Weekly trend filter ensures we only trade with the higher timeframe momentum, reducing whipsaws in ranging markets.
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.
# Works in bull markets by catching dips in uptrends and in bear markets by selling rallies in downtrends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Weekly EMA200 for trend filter (loaded once before loop)
    df_1w = get_htf_data(prices, '1w')
    ema200_1w = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Daily EMA50 for dynamic support/resistance
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily RSI(14) for entry timing
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for EMA50 and RSI
    
    for i in range(start_idx, n):
        if (np.isnan(ema50[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema50_val = ema50[i]
        rsi_val = rsi[i]
        weekly_trend = ema200_1w_aligned[i]  # Weekly EMA200 value
        
        if position == 0:
            # Long: Weekly uptrend (price > weekly EMA200) + price near EMA50 + RSI oversold
            if price > weekly_trend and price <= ema50_val * 1.02 and rsi_val < 40:
                signals[i] = 0.25
                position = 1
            # Short: Weekly downtrend (price < weekly EMA200) + price near EMA50 + RSI overbought
            elif price < weekly_trend and price >= ema50_val * 0.98 and rsi_val > 60:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: RSI overbought or price moves significantly above EMA50
            if rsi_val > 70 or price > ema50_val * 1.05:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: RSI oversold or price moves significantly below EMA50
            if rsi_val < 30 or price < ema50_val * 0.95:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WeeklyTrend_RSI_Pullback"
timeframe = "1d"
leverage = 1.0