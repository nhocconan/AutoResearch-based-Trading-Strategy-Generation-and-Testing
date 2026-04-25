#!/usr/bin/env python3
"""
1d_KAMA_RSI_ChopFilter_v1
Hypothesis: Trade KAMA trend direction on 1d with RSI momentum filter and chop regime filter.
KAMA adapts to market noise, reducing whipsaws in choppy markets. RSI>55 confirms bullish momentum,
RSI<45 confirms bearish momentum. Chop filter (EWMA of |close - open| / ATR) avoids ranging markets.
This combination should work in both bull and bear markets by filtering false signals and capturing
strong trending moves with volume confirmation. Target: 7-25 trades/year per symbol.
"""

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
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA20 for HTF trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # KAMA (adaptive moving average) on 1d
    # Efficiency Ratio: ER = |close - close[10]| / sum(|close - close[1]|, 10)
    close_s = pd.Series(close)
    change = abs(close_s.diff(10)).values
    volatility = close_s.diff().abs().rolling(window=10, min_periods=10).sum().values
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    # Smoothing constants: fastest SC=2/(2+1)=0.667, slowest SC=2/(30+1)=0.0645
    sc = (er * 0.602 + 0.0645) ** 2  # ER*(0.667-0.0645)+0.0645, squared for smoother adaptation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14) for momentum confirmation
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Chop filter: EWMA of |close - open| / ATR(14) to detect ranging markets
    true_range = np.maximum(high - low, np.maximum(abs(high - close_s.shift()), abs(low - close_s.shift())))
    atr = pd.Series(true_range).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    body_ratio = abs(close - prices['open'].values) / atr
    chop = pd.Series(body_ratio).ewm(alpha=1/10, adjust=False).mean().values
    
    # Volume confirmation: 1d volume > 1.3 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for KAMA (10), EMA20_1w (20), RSI (14), ATR (14), volume MA (20)
    start_idx = max(10, 20, 14, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1w HTF trend (bullish = price above EMA20)
        htf_1w_bullish = close[i] > ema_20_1w_aligned[i]
        htf_1w_bearish = close[i] < ema_20_1w_aligned[i]
        
        if position == 0:
            # Long setup: price > KAMA (bullish trend) + RSI > 55 (momentum) + chop < 0.4 (trending) + volume spike
            long_setup = (close[i] > kama[i]) and (rsi[i] > 55) and (chop[i] < 0.4) and volume_spike[i]
            
            # Short setup: price < KAMA (bearish trend) + RSI < 45 (momentum) + chop < 0.4 (trending) + volume spike
            short_setup = (close[i] < kama[i]) and (rsi[i] < 45) and (chop[i] < 0.4) and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price < KAMA (trend change) OR RSI < 40 (momentum loss) OR chop > 0.5 (ranging)
            if (close[i] < kama[i]) or (rsi[i] < 40) or (chop[i] > 0.5):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price > KAMA (trend change) OR RSI > 60 (momentum loss) OR chop > 0.5 (ranging)
            if (close[i] > kama[i]) or (rsi[i] > 60) or (chop[i] > 0.5):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0