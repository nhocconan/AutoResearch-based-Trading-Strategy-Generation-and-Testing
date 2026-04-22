# 4h_200EMA_RSI2080_TrendFollowing_v1
# Hypothesis: Use 200EMA as long-term trend filter (works in bull/bear) with RSI(14) for momentum entries (20/80 levels).
# Only trade in direction of 200EMA to avoid counter-trend whipsaws. Volume confirmation filters low-conviction moves.
# Target: 20-40 trades/year on 4h timeframe with clear trend + momentum signals.
# Should work in bull (trend up + RSI buy signals) and bear (trend down + RSI sell signals).

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # 1h trend filter (HTF) - more responsive than daily for 4h signals
    df_1h = get_htf_data(prices, '1h')
    close_1h = df_1h['close'].values
    ema_200_1h = pd.Series(close_1h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_200_1h)
    
    # 4h indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # Volume average for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(ema_200_1h_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_trend = ema_200_1h_aligned[i]
        rsi_val = rsi[i]
        vol = volume[i]
        vol_avg_val = vol_avg[i]
        
        # Volume confirmation: above average volume
        vol_confirm = vol > vol_avg_val
        
        if position == 0:
            # Long: above 200EMA (uptrend) + RSI oversold bounce + volume
            if price > ema_trend and rsi_val < 30 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: below 200EMA (downtrend) + RSI overbought bounce + volume
            elif price < ema_trend and rsi_val > 70 and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit: RSI overbought or trend break
            if rsi_val > 70 or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # hold
        
        elif position == -1:  # Short position
            # Exit: RSI oversold or trend break
            if rsi_val < 30 or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # hold
    
    return signals

name = "4h_200EMA_RSI2080_TrendFollowing_v1"
timeframe = "4h"
leverage = 1.0