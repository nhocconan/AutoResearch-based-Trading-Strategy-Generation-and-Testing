#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h trend filter and 1d volatility regime filter.
# Long when: 1h RSI(14) > 55, 4h EMA(20) > EMA(50) (uptrend), and 1d ATR(14) < ATR(50) (low vol regime)
# Short when: 1h RSI(14) < 45, 4h EMA(20) < EMA(50) (downtrend), and 1d ATR(14) < ATR(50)
# Exit when RSI crosses back to 50 or volatility regime changes
# Uses 4h EMA for trend direction, 1h RSI for entry timing, 1d ATR regime to avoid choppy markets.
# Target: 15-30 trades/year per symbol (~60-120 total over 4 years) to stay within fee limits.
name = "1h_RSI_EMA_Trend_VolRegime"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 4h EMA for trend filter
    df_4h = get_htf_data(prices, '4h')
    ema20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Get 1d ATR for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50_1d = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    atr50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr50_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 20, 50)  # RSI(14), EMA20, EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi[i]) or np.isnan(ema20_4h_aligned[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(atr14_1d_aligned[i]) or 
            np.isnan(atr50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        rsi_val = rsi[i]
        ema20 = ema20_4h_aligned[i]
        ema50 = ema50_4h_aligned[i]
        atr14 = atr14_1d_aligned[i]
        atr50 = atr50_1d_aligned[i]
        hour = hours[i]
        
        # Regime filters: trend and low volatility
        uptrend = ema20 > ema50
        downtrend = ema20 < ema50
        low_vol = atr14 < atr50
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            # Long entry: RSI > 55, uptrend, low vol, in session
            if rsi_val > 55 and uptrend and low_vol and in_session:
                signals[i] = 0.20
                position = 1
            # Short entry: RSI < 45, downtrend, low vol, in session
            elif rsi_val < 45 and downtrend and low_vol and in_session:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: RSI < 50 or trend breaks or high vol or outside session
            if rsi_val < 50 or not uptrend or not low_vol or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: RSI > 50 or trend breaks or high vol or outside session
            if rsi_val > 50 or not downtrend or not low_vol or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals