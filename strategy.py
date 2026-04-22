#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum strategy with 4h/1d trend filter and volume confirmation.
# Uses 4h RSI(14) for momentum, 1d EMA(50) for trend, and volume spike for confirmation.
# Long when 4h RSI crosses above 50 in uptrend (close > daily EMA50) with volume spike.
# Short when 4h RSI crosses below 50 in downtrend (close < daily EMA50) with volume spike.
# Exit on opposite RSI cross or trend reversal.
# Designed for 1h timeframe to target 15-37 trades/year per symbol.
# Uses 4h/1d for signal direction, 1h only for entry timing.
# Session filter (08-20 UTC) to reduce noise trades.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for RSI (ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Load 1d data for trend (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 4h RSI(14)
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs))
    
    # Calculate 1d EMA(50) for trend
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 1h timeframe
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma20[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI crosses above 50 + uptrend (close > EMA50) + volume spike
            if (rsi_4h_aligned[i] > 50 and rsi_4h_aligned[i-1] <= 50 and
                close[i] > ema_50_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: RSI crosses below 50 + downtrend (close < EMA50) + volume spike
            elif (rsi_4h_aligned[i] < 50 and rsi_4h_aligned[i-1] >= 50 and
                  close[i] < ema_50_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit on RSI cross below 50 or trend reversal
                if (rsi_4h_aligned[i] < 50 and rsi_4h_aligned[i-1] >= 50) or close[i] < ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                # Exit on RSI cross above 50 or trend reversal
                if (rsi_4h_aligned[i] > 50 and rsi_4h_aligned[i-1] <= 50) or close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_RSI50_4hMom_1dTrend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0