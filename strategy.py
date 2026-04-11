#!/usr/bin/env python3
"""
1d_1w_rsi_reversion_volume
Strategy: 1d RSI mean reversion with volume confirmation and weekly trend filter
Timeframe: 1d
Leverage: 1.0
Hypothesis: Uses daily RSI(14) for mean reversion entries (long when RSI<30, short when RSI>70) with volume confirmation (>1.5x average volume) and filtered by weekly EMA50 trend. Designed to capture reversals in both bull and bear markets by combining oversold/overbought conditions with trend alignment. Weekly EMA50 ensures we only trade in the direction of the higher timeframe trend, reducing false signals in choppy markets. Target: 20-60 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_rsi_reversion_volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Daily RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if any required data is invalid
        if (np.isnan(rsi[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below weekly EMA50
        uptrend_1w = price_close > ema_50_1w_aligned[i]
        downtrend_1w = price_close < ema_50_1w_aligned[i]
        
        # RSI mean reversion conditions
        oversold = rsi[i] < 30
        overbought = rsi[i] > 70
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: oversold RSI with volume in uptrend
        long_signal = oversold and vol_confirmed and uptrend_1w
        
        # Short: overbought RSI with volume in downtrend
        short_signal = overbought and vol_confirmed and downtrend_1w
        
        # Exit when RSI returns to neutral zone (40-60) or opposite extreme
        exit_long = position == 1 and (rsi[i] > 40 or rsi[i] > 60)
        exit_short = position == -1 and (rsi[i] < 60 or rsi[i] < 40)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals