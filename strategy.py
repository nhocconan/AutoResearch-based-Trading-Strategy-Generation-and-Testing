#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for primary trend
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h EMA200 for trend filter
    close_4h_series = pd.Series(close_4h)
    ema200_4h = close_4h_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # Calculate 4h ATR(14) for volatility filter
    tr4h = np.maximum(high_4h - low_4h, np.maximum(np.abs(high_4h - np.roll(close_4h, 1)), np.abs(low_4h - np.roll(close_4h, 1))))
    tr4h[0] = high_4h[0] - low_4h[0]
    atr4h = pd.Series(tr4h).rolling(window=14, min_periods=14).mean().values
    atr4h_aligned = align_htf_to_ltf(prices, df_4h, atr4h)
    
    # Get 1h ATR(14) for entry conditions
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1h RSI(7) for mean reversion entries
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/7, adjust=False, min_periods=7).mean()
    avg_loss = loss.ewm(alpha=1/7, adjust=False, min_periods=7).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Calculate 1h Bollinger Bands for volatility breakout
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean()
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std()
    upper_bb = sma20 + (2 * std20)
    lower_bb = sma20 - (2 * std20)
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(200, 20)  # Need EMA200 warmup
    
    for i in range(start_idx, n):
        if not session_filter[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is not available
        if (np.isnan(ema200_4h_aligned[i]) or np.isnan(atr4h_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(upper_bb[i]) or np.isnan(lower_bb[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 4h EMA200
        uptrend = close[i] > ema200_4h_aligned[i]
        downtrend = close[i] < ema200_4h_aligned[i]
        
        # Volatility filter: current ATR > 0.5 * 4h ATR (avoid low volatility chop)
        vol_filter = atr[i] > (0.5 * atr4h_aligned[i])
        
        if position == 0:
            # Long entry: pullback in uptrend (RSI < 30) OR breakout above upper BB with volume
            if uptrend and vol_filter:
                if rsi[i] < 30:  # Oversold pullback in uptrend
                    signals[i] = 0.20
                    position = 1
                elif close[i] > upper_bb[i]:  # Breakout above BB
                    signals[i] = 0.20
                    position = 1
            # Short entry: pullback in downtrend (RSI > 70) OR breakdown below lower BB
            elif downtrend and vol_filter:
                if rsi[i] > 70:  # Overbought pullback in downtrend
                    signals[i] = -0.20
                    position = -1
                elif close[i] < lower_bb[i]:  # Breakdown below BB
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:
            # Exit long: RSI > 70 (overbought) or price < 4h EMA200 (trend change)
            if rsi[i] > 70 or close[i] < ema200_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: RSI < 30 (oversold) or price > 4h EMA200 (trend change)
            if rsi[i] < 30 or close[i] > ema200_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA200_RSI_BB_Pullback_Breakout"
timeframe = "1h"
leverage = 1.0