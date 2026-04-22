#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI with 4h trend filter and volume confirmation.
# Uses 4h EMA(50) for trend direction, 1h RSI(14) for entry timing with overbought/oversold thresholds.
# Volume spike filter reduces false signals.
# Long in uptrend when RSI < 30 (oversold) + volume spike.
# Short in downtrend when RSI > 70 (overbought) + volume spike.
# Session filter (08-20 UTC) to avoid low-liquidity hours.
# Target: 15-37 trades/year per symbol (60-150 total) to stay within fee limits.
# Designed to work in both bull and bear markets via trend-following + mean-reversion entries.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data for trend filter (ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 4h EMA(50) for trend direction
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if not in session or data not ready
        if not in_session[i] or np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: uptrend (price > EMA50) + RSI oversold + volume spike
            if (close[i] > ema_50_4h_aligned[i] and 
                rsi[i] < 30 and 
                vol_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: downtrend (price < EMA50) + RSI overbought + volume spike
            elif (close[i] < ema_50_4h_aligned[i] and 
                  rsi[i] > 70 and 
                  vol_spike[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit: trend reversal or RSI mean reversion
            if position == 1:
                if (close[i] < ema_50_4h_aligned[i] or rsi[i] > 50):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if (close[i] > ema_50_4h_aligned[i] or rsi[i] < 50):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_RSI14_4hEMA50_Trend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0