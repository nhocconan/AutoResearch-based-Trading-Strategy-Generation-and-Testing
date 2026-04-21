#!/usr/bin/env python3
"""
1d_RSI_Withdrawal_V1
Hypothesis: Buy when daily RSI drops below 30 (oversold) with volume confirmation,
sell when RSI rises above 70 (overbought). Uses weekly trend filter to avoid
counter-trend trades in strong trends. Designed for low frequency (5-15 trades/year)
to minimize fee drag and work in both bull and bear markets via mean reversion
within the prevailing weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate RSI(14) on daily
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.values
    
    # Align RSI to lower timeframe (no additional delay needed for RSI)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Load weekly data for trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(rsi_1d_aligned[i]) or np.isnan(ema_34_1w_aligned[i]):
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-day average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        rsi = rsi_1d_aligned[i]
        weekly_ema = ema_34_1w_aligned[i]
        
        # Long: RSI oversold (<30) in uptrend (price > weekly EMA) with volume
        if rsi < 30 and price > weekly_ema and volume_ok:
            signals[i] = 0.25
        # Short: RSI overbought (>70) in downtrend (price < weekly EMA) with volume
        elif rsi > 70 and price < weekly_ema and volume_ok:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_RSI_Withdrawal_V1"
timeframe = "1d"
leverage = 1.0