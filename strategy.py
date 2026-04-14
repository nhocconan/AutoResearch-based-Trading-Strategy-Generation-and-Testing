#!/usr/bin/env python3
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA(21) for trend filter
    ema_21_1w = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate weekly ATR(14) for volatility filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate weekly RSI(14) for momentum filter
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 100
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_21_1w_aligned[i]) or 
            np.isnan(atr_1w_aligned[i]) or
            np.isnan(rsi_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: price > weekly EMA21 for long, price < weekly EMA21 for short
        trend_filter_long = price > ema_21_1w_aligned[i]
        trend_filter_short = price < ema_21_1w_aligned[i]
        
        # Volatility filter: weekly ATR > 1.5% of price to avoid low volatility periods
        vol_filter = atr_1w_aligned[i] / price > 0.015 if price > 0 else False
        
        # Momentum filter: RSI between 30 and 70 to avoid extremes
        mom_filter = (rsi_1w_aligned[i] >= 30) & (rsi_1w_aligned[i] <= 70)
        
        if position == 0:
            # Long setup: price above weekly EMA21 + volatility filter + momentum filter
            if trend_filter_long and vol_filter and mom_filter:
                position = 1
                signals[i] = position_size
            # Short setup: price below weekly EMA21 + volatility filter + momentum filter
            elif trend_filter_short and vol_filter and mom_filter:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below weekly EMA21
            if price < ema_21_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above weekly EMA21
            if price > ema_21_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_weekly_EMA21_VolFilter_RSI_Momentum_v1"
timeframe = "1d"
leverage = 1.0