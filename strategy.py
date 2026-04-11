#!/usr/bin/env python3
# 1d_1w_rsi_volume_extreme_v1
# Strategy: 1d RSI extremes with volume confirmation and weekly trend filter
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: RSI extremes (<30 or >70) indicate overextended conditions likely to reverse.
# Volume confirmation ensures institutional participation. Weekly trend filter (EMA50) ensures
# trades align with higher-timeframe momentum, reducing counter-trend trades in strong trends.
# Designed for low trade frequency (~10-25/year) to minimize fee drag. Works in bull markets
# by buying oversold dips in uptrends and selling overbought rallies in downtrends. In ranging
# markets, captures mean reversion at extremes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_rsi_volume_extreme_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1d RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(rsi[i]) or np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_avg_20[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # RSI conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Weekly trend: price above/below EMA50
        price_above_weekly_ema = close[i] > ema_50_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_50_1w_aligned[i]
        
        # Entry conditions
        # Long: RSI oversold AND price above weekly EMA (uptrend) AND volume confirmation
        if rsi_oversold and price_above_weekly_ema and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: RSI overbought AND price below weekly EMA (downtrend) AND volume confirmation
        elif rsi_overbought and price_below_weekly_ema and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: RSI returns to neutral zone (40-60)
        elif position == 1 and rsi[i] >= 40:
            position = 0
            signals[i] = 0.0
        elif position == -1 and rsi[i] <= 60:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals