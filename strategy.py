#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) pullback with 4h trend filter and 1d volume confirmation
# RSI < 30 in uptrend (price > 4h EMA50) for long, RSI > 70 in downtrend for short
# Uses 1d volume spike to filter for institutional participation
# Session filter (08-20 UTC) reduces noise
# Target: 15-30 trades/year per symbol (60-120 total) to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate RSI(14) on close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ema = pd.Series(gain).ewm(alpha=1/14, adjust=False).values
    loss_ema = pd.Series(loss).ewm(alpha=1/14, adjust=False).values
    rs = gain_ema / (loss_ema + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Load 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Load 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > 1.5 * vol_ma20_1d
    
    # Align indicators to 1-hour timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(rsi[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_spike_1d_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold) + uptrend (price > 4h EMA50) + 1d volume spike
            if (rsi[i] < 30 and close[i] > ema_50_4h_aligned[i] and vol_spike_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: RSI > 70 (overbought) + downtrend (price < 4h EMA50) + 1d volume spike
            elif (rsi[i] > 70 and close[i] < ema_50_4h_aligned[i] and vol_spike_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit: RSI returns to neutral zone (40-60)
            if position == 1:
                if rsi[i] > 40:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if rsi[i] < 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_RSI_Pullback_4hEMA50_1dVolume_Spike_Session"
timeframe = "1h"
leverage = 1.0