#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum strategy with 4h trend filter and session filter
# - Use 4h EMA(50) for trend direction (long when price > EMA, short when price < EMA)
# - Enter on 1h RSI(14) pullbacks in trending direction (RSI < 40 for long, RSI > 60 for short)
# - Add volume confirmation: current volume > 1.2x 20-period average
# - Time filter: only trade 08-20 UTC to avoid low-volume Asian session
# - Fixed position size: 0.20 (20% of capital)
# - Exit on opposite RSI extreme or trend reversal
# - Target: 20-40 trades per year per symbol (80-160 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA(50) on 4h close
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1h data for entry signals
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate RSI(14) on 1h close
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        ema_trend = ema_50_4h_aligned[i]
        
        if position == 0:
            # Long entry: uptrend + RSI pullback + volume
            if price > ema_trend and rsi[i] < 40 and vol > 1.2 * vol_ma[i]:
                signals[i] = 0.20
                position = 1
            # Short entry: downtrend + RSI bounce + volume
            elif price < ema_trend and rsi[i] > 60 and vol > 1.2 * vol_ma[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: RSI overbought OR trend reversal
            if rsi[i] > 60 or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: RSI oversold OR trend reversal
            if rsi[i] < 40 or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA50_RSI_Pullback_Volume_Session"
timeframe = "1h"
leverage = 1.0