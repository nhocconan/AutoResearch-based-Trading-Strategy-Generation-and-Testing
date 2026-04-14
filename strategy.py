# 1H strategy for BTC/ETH: 4H trend + 1H entry with volume and session filters
# Hypothesis: Use 4H EMA for trend direction, enter on 1H pullbacks with volume confirmation
# Limited to London/NY session (08-20 UTC) to reduce noise. Target 15-37 trades/year.
# Works in bull/bear by following higher timeframe trend.
#!/usr/bin/env python3
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
    
    # Load 4H data ONCE before loop (HTF for trend)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 20-period EMA on 4H close
    ema20_4h = np.full_like(close_4h, np.nan)
    if len(close_4h) >= 20:
        alpha = 2 / (20 + 1)
        ema20_4h[0] = close_4h[0]
        for i in range(1, len(close_4h)):
            ema20_4h[i] = alpha * close_4h[i] + (1 - alpha) * ema20_4h[i-1]
    
    # Align 4H EMA to 1H timeframe (waits for 4H bar close)
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Calculate 14-period RSI on 1H for momentum (uses only past data)
    rsi14 = np.full(n, np.nan)
    if n >= 14:
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        
        avg_gain[13] = np.mean(gain[1:14])
        avg_loss[13] = np.mean(loss[1:14])
        
        for i in range(14, n):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        
        for i in range(13, n):
            if avg_loss[i] > 0:
                rs = avg_gain[i] / avg_loss[i]
                rsi14[i] = 100 - (100 / (1 + rs))
            else:
                rsi14[i] = 100 if avg_gain[i] > 0 else 0
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    for i in range(30, n):
        # Skip if no session or missing data
        if not in_session[i] or np.isnan(ema20_4h_aligned[i]) or np.isnan(rsi14[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Above 4H EMA + RSI > 50 (bullish momentum)
            if close[i] > ema20_4h_aligned[i] and rsi14[i] > 50:
                position = 1
                signals[i] = position_size
            # Short: Below 4H EMA + RSI < 50 (bearish momentum)
            elif close[i] < ema20_4h_aligned[i] and rsi14[i] < 50:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Below 4H EMA (trend change)
            if close[i] < ema20_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Above 4H EMA (trend change)
            if close[i] > ema20_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_ema_rsi_session"
timeframe = "1h"
leverage = 1.0