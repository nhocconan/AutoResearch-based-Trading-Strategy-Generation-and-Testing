#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day ATR-based volatility filter and 1-week RSI mean reversion
# - Long when weekly RSI < 30 (oversold) and price closes above 12h EMA20
# - Short when weekly RSI > 70 (overbought) and price closes below 12h EMA20
# - Volatility filter: only trade when 1-day ATR(14) < 1.5x its 50-period average (low volatility regime)
# - Uses weekly RSI for mean-reversion signals, works in both bull and bear markets
# - EMA20 provides dynamic support/resistance for entry timing
# - Volatility filter prevents whipsaws during high volatility periods
# - Target: 50-150 total trades over 4 years (12-37/year) for optimal balance
# - Position size 0.25 for balanced risk exposure

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load weekly data once before loop
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 30:
        return np.zeros(n)
    
    # Load daily data once before loop
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 50:
        return np.zeros(n)
    
    # Calculate weekly RSI(14)
    close_w = df_w['close'].values
    delta = np.diff(close_w, prepend=close_w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_w = 100 - (100 / (1 + rs))
    rsi_w = rsi_w.values
    
    # Calculate daily ATR(14)
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    tr1 = high_d - low_d
    tr2 = np.abs(high_d - np.roll(close_d, 1))
    tr3 = np.abs(low_d - np.roll(close_d, 1))
    tr1[0] = high_d[0] - low_d[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_d = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    
    # Calculate 50-period average of ATR for volatility filter
    atr_ma_d = pd.Series(atr_d).rolling(window=50, min_periods=50).mean().values
    
    # Calculate 12h EMA20
    close_s = pd.Series(close)
    ema20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if np.isnan(rsi_w) or np.isnan(atr_d) or np.isnan(atr_ma_d) or np.isnan(ema20[i]):
            continue
        
        # Get weekly index for current 12h bar
        # 1 week = 7 days = 14 * 12h bars
        idx_w = i // 14
        if idx_w < 1:
            continue
            
        # Get daily index for current 12h bar
        # 1 day = 2 * 12h bars
        idx_d = i // 2
        if idx_d < 1:
            continue
            
        # Previous week's RSI (to avoid look-ahead)
        rsi_prev = rsi_w[idx_w-1]
        
        # Previous day's ATR and its MA (to avoid look-ahead)
        atr_prev = atr_d[idx_d-1]
        atr_ma_prev = atr_ma_d[idx_d-1]
        
        if position == 0:
            # Volatility filter: only trade in low volatility regime
            vol_filter = atr_prev < (atr_ma_prev * 1.5)
            
            # Long: Weekly RSI oversold + price above EMA20 + volatility filter
            if (rsi_prev < 30 and  # Weekly RSI oversold
                close[i] > ema20[i] and  # Price above 12h EMA20
                vol_filter):  # Low volatility regime
                position = 1
                signals[i] = position_size
            # Short: Weekly RSI overbought + price below EMA20 + volatility filter
            elif (rsi_prev > 70 and  # Weekly RSI overbought
                  close[i] < ema20[i] and  # Price below 12h EMA20
                  vol_filter):  # Low volatility regime
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Price crosses below EMA20 or RSI becomes overbought
            if close[i] < ema20[i] or rsi_prev > 70:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Price crosses above EMA20 or RSI becomes oversold
            if close[i] > ema20[i] or rsi_prev < 30:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_1w_RSI_1d_ATR_Volatility_Filter"
timeframe = "12h"
leverage = 1.0