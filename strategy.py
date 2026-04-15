#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) + 4h/1d Trend Filter + Session Filter
# Uses RSI for mean reversion entries during high-probability sessions (08-20 UTC).
# Long when RSI < 30 and price > 4h EMA200 (bull trend filter); short when RSI > 70 and price < 4h EMA200 (bear trend filter).
# Uses 1d ADX > 25 to filter ranging markets and avoid false signals.
# Designed for low trade frequency (<30/year) with high win rate in both bull/bear markets via trend alignment.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (precomputed for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h EMA200 for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d ADX(14) for trend strength filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).diff()
    tr2 = pd.Series(high_1d).sub(pd.Series(close_1d).shift())
    tr3 = pd.Series(low_1d).sub(pd.Series(close_1d).shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up = pd.Series(high_1d).diff()
    down = pd.Series(low_1d).diff()
    up = np.where((up > down) & (up > 0), up, 0)
    down = np.where((down > up) & (down > 0), down, 0)
    
    # Smoothed DM
    up_smooth = pd.Series(up).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    down_smooth = pd.Series(down).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # DI and DX
    di_plus = 100 * up_smooth / atr_1d
    di_minus = 100 * down_smooth / atr_1d
    dx = (abs(di_plus - di_minus) / (di_plus + di_minus)) * 100
    adx_1d = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d.values)
    
    # RSI(14) on 1h close
    delta = pd.Series(close).diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    signals = np.zeros(n)
    
    for i in range(14, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or not in_session[i]):
            continue
        
        # Long: RSI oversold, price above 4h EMA200, strong trend (ADX > 25)
        if (rsi[i] < 30 and 
            close[i] > ema_4h_aligned[i] and 
            adx_1d_aligned[i] > 25):
            signals[i] = 0.20
        
        # Short: RSI overbought, price below 4h EMA200, strong trend (ADX > 25)
        elif (rsi[i] > 70 and 
              close[i] < ema_4h_aligned[i] and 
              adx_1d_aligned[i] > 25):
            signals[i] = -0.20
        
        # Exit: RSI returns to neutral zone (40-60)
        elif (i > 0 and 
              ((signals[i-1] == 0.20 and rsi[i] >= 40) or
               (signals[i-1] == -0.20 and rsi[i] <= 60))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1h_RSI_4hEMA200_1dADX_Session"
timeframe = "1h"
leverage = 1.0