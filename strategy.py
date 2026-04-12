# 1d_1w_rsi_momentum_trend_v1
# Hypothesis: Daily RSI momentum filtered by weekly trend and volatility regime.
# In bull markets, RSI > 60 with weekly uptrend captures momentum.
# In bear markets, RSI < 40 with weekly downtrend captures mean-reversion bounces.
# Weekly trend filter reduces whipsaws; volatility filter avoids chop.
# Target: 15-25 trades/year, discretionary sizing 0.25.

#!/usr/bin/env python3
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
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly 20-period EMA for trend
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Daily 14-period RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily ATR(10) for volatility filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr10 = np.full(n, np.nan)
    for i in range(9, n):
        atr10[i] = np.nanmean(tr[i-9:i+1])
    
    # ATR(10) EMA(20) for volatility regime
    atr_ema20 = np.full(n, np.nan)
    atr_series = pd.Series(atr10)
    atr_ema20_values = atr_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_ema20[:] = atr_ema20_values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # warmup for RSI and ATR
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(atr10[i]) or np.isnan(atr_ema20[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema20_1w_aligned[i]
        weekly_downtrend = close[i] < ema20_1w_aligned[i]
        
        # Volatility filter: elevated volatility (avoid chop)
        vol_filter = atr10[i] > atr_ema20[i] * 0.8
        
        # RSI momentum conditions
        rsi_overbought = rsi[i] > 60
        rsi_oversold = rsi[i] < 40
        
        # Entry logic: align RSI momentum with weekly trend
        long_entry = rsi_overbought and weekly_uptrend and vol_filter
        short_entry = rsi_oversold and weekly_downtrend and vol_filter
        
        # Exit on RSI mean reversion or volatility collapse
        long_exit = rsi[i] < 50 or atr10[i] < atr_ema20[i] * 0.6
        short_exit = rsi[i] > 50 or atr10[i] < atr_ema20[i] * 0.6
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_rsi_momentum_trend_v1"
timeframe = "1d"
leverage = 1.0