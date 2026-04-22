#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h momentum with 4h trend filter and volume confirmation
    # Uses RSI(14) for momentum, 4h EMA50 for trend direction, and volume spike for confirmation
    # Works in bull markets via momentum continuation and in bear via mean reversion from oversold/overbought
    # Target: 15-37 trades/year (~60-150 total over 4 years) to avoid fee drag
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for trend filter (higher timeframe)
    df_4h = get_htf_data(prices, '4h')
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # RSI(14) momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready or outside session
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI > 55 (bullish momentum) + price above 4h EMA50 + volume spike
            if rsi[i] > 55 and close[i] > ema50_4h_aligned[i] and vol_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: RSI < 45 (bearish momentum) + price below 4h EMA50 + volume spike
            elif rsi[i] < 45 and close[i] < ema50_4h_aligned[i] and vol_spike[i]:
                signals[i] = -0.20
                position = -1
        else:
            # Exit: RSI crosses 50 (momentum shift) or price crosses 4h EMA50 (trend change)
            if position == 1:
                if rsi[i] < 50 or close[i] < ema50_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if rsi[i] > 50 or close[i] > ema50_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_RSI_Momentum_4hEMA50_Trend_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0