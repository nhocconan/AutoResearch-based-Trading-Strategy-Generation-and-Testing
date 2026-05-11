#!/usr/bin/env python3
name = "1d_KAMA_Trend_RSI_Pullback_TrendFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d KAMA for trend
    close_series = pd.Series(close)
    delta = close_series.diff().abs()
    vol = delta.rolling(window=10, min_periods=10).sum()
    er = delta.rolling(window=10, min_periods=10).sum() / vol.replace(0, np.nan)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate 1w trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate RSI(14)
    delta_rsi = pd.Series(close).diff()
    gain = delta_rsi.where(delta_rsi > 0, 0)
    loss = -delta_rsi.where(delta_rsi < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume filter: 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        if position == 0:
            # Long: Price above KAMA + RSI oversold + weekly uptrend + volume
            if price_above_kama and rsi_oversold and close[i] > ema34_1w_aligned[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA + RSI overbought + weekly downtrend + volume
            elif price_below_kama and rsi_overbought and close[i] < ema34_1w_aligned[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit: Price crosses below KAMA OR RSI overbought
                if close[i] < kama[i] or rsi[i] > 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit: Price crosses above KAMA OR RSI oversold
                if close[i] > kama[i] or rsi[i] < 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals