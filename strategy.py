#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_trend_pullback_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Daily EMA trend filter
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h ATR for volatility and stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h RSI for pullback entries
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_20_aligned[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_low = low[i]
        price_high = high[i]
        ema_20 = ema_20_aligned[i]
        ema_50 = ema_50_aligned[i]
        
        # Trend condition: EMA20 > EMA50 for uptrend, EMA20 < EMA50 for downtrend
        uptrend = ema_20 > ema_50
        downtrend = ema_20 < ema_50
        
        # Pullback entry conditions
        long_entry = False
        short_entry = False
        
        # Long: Uptrend + price pulls back to EMA20 with RSI < 40
        if uptrend and price_low <= ema_20 and rsi[i] < 40:
            long_entry = True
        
        # Short: Downtrend + price pulls back to EMA20 with RSI > 60
        if downtrend and price_high >= ema_20 and rsi[i] > 60:
            short_entry = True
        
        # Exit conditions
        # Exit long when price closes above EMA50 (trend weakening) or RSI > 70
        exit_long = position == 1 and (price_close > ema_50 or rsi[i] > 70)
        # Exit short when price closes below EMA50 or RSI < 30
        exit_short = position == -1 and (price_close < ema_50 or rsi[i] < 30)
        
        # Stop loss: 2.5 * ATR from entry
        stop_long = position == 1 and price_low < (entry_price - 2.5 * atr[i])
        stop_short = position == -1 and price_high > (entry_price + 2.5 * atr[i])
        
        # Trading logic
        if long_entry and position != 1:
            position = 1
            entry_price = price_close
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            entry_price = price_close
            signals[i] = -0.25
        elif position == 1 and (exit_long or stop_long):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (exit_short or stop_short):
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Trend pullback strategy for 4h timeframe using daily EMA trend filter and 4h RSI for entry timing.
# In uptrends (daily EMA20 > EMA50), enter long when price pulls back to EMA20 with RSI < 40 (oversold).
# In downtrends (daily EMA20 < EMA50), enter short when price pulls back to EMA20 with RSI > 60 (overbought).
# Exits when trend weakens (price crosses EMA50) or RSI reaches extreme levels.
# Stop loss at 2.5 * ATR to manage risk.
# Designed for low trade frequency (~20-40 trades/year) to minimize fee drag.
# Works in both bull and bear markets by following the trend on higher timeframe.
# Uses daily EMA for trend (avoids 4h whipsaws) and 4h RSI for precise entry timing.
# Discrete position sizing (0.25) to reduce churn from small signal changes.
# Should generate 80-160 total trades over 4 years (20-40/year) based on similar strategies.
# Works on BTC/ETH/SOL by focusing on trend continuation after pullbacks.