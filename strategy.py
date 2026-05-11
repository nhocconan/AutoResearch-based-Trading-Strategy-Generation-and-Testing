#!/usr/bin/env python3
name = "1h_Liquidity_Rebalance_4hTrend"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend and liquidity imbalance
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h EMA 20 for trend
    ema20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # 4h volume imbalance: (close - open) / (high - low) * volume
    # Normalized to [-1, 1], positive = buying pressure
    price_range_4h = df_4h['high'].values - df_4h['low'].values
    price_range_4h = np.where(price_range_4h == 0, 1, price_range_4h)  # avoid div by zero
    vol_imbalance_4h = ((df_4h['close'].values - df_4h['open'].values) / price_range_4h) * df_4h['volume'].values
    vol_imbalance_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_imbalance_4h)
    
    # 1h RSI(14) for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss == 0, 0, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    # 1h volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(vol_imbalance_4h_aligned[i]) or
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Only trade during session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: 4h uptrend (price > EMA20) AND bullish volume imbalance AND RSI < 40 (oversold) AND volume spike
            if (close[i] > ema20_4h_aligned[i] and 
                vol_imbalance_4h_aligned[i] > 0.1 and 
                rsi[i] < 40 and 
                vol_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend (price < EMA20) AND bearish volume imbalance AND RSI > 60 (overbought) AND volume spike
            elif (close[i] < ema20_4h_aligned[i] and 
                  vol_imbalance_4h_aligned[i] < -0.1 and 
                  rsi[i] > 60 and 
                  vol_spike[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: 4h trend turns down OR RSI > 70 (overbought) OR volume imbalance turns bearish
            if (close[i] < ema20_4h_aligned[i] or 
                rsi[i] > 70 or 
                vol_imbalance_4h_aligned[i] < -0.1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20  # maintain position
        elif position == -1:
            # Short exit: 4h trend turns up OR RSI < 30 (oversold) OR volume imbalance turns bullish
            if (close[i] > ema20_4h_aligned[i] or 
                rsi[i] < 30 or 
                vol_imbalance_4h_aligned[i] > 0.1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20  # maintain position
    
    return signals