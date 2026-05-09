# 1h_MomentumReversal_VolumeSpike_4hTrend
# Strategy type: 4h trend direction with 1h momentum reversal entries
# Rationale: In strong 4h trends, 1h pullbacks with volume spikes offer high-probability entries.
# Works in bull/bear by following the 4h trend direction, avoiding counter-trend trades.
# Volume spike filters out low-conviction moves. Session filter (08-20 UTC) reduces noise.
# Target: 15-30 trades/year per symbol by requiring 4h trend alignment + volume spike + momentum reversal.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_MomentumReversal_VolumeSpike_4hTrend"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1h = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # RSI(14) for momentum reversal signal
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume filter: spike above 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Wait for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1h[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma[i]  # Volume spike confirmation
        
        # Pre-compute hour for session filter (UTC 8-20)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long: 4h uptrend (price > EMA50) + RSI < 30 (oversold) + volume spike + session
            if (close[i] > ema_50_1h[i] and 
                rsi[i] < 30 and 
                vol_ok and 
                in_session):
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend (price < EMA50) + RSI > 70 (overbought) + volume spike + session
            elif (close[i] < ema_50_1h[i] and 
                  rsi[i] > 70 and 
                  vol_ok and 
                  in_session):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: RSI > 50 (momentum recovered) or trend reversal
            if rsi[i] > 50 or close[i] < ema_50_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: RSI < 50 (momentum recovered) or trend reversal
            if rsi[i] < 50 or close[i] > ema_50_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals