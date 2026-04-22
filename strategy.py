#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h momentum with 4h trend filter and volume confirmation
    # Uses 1h RSI(14) for momentum timing, 4h EMA50 for trend direction,
    # and volume spike (2x 20-period MA) for institutional confirmation
    # Designed to work in both bull and bear markets by requiring alignment
    # between short-term momentum and long-term trend with volume validation
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 8-20 UTC (pre-loaded for efficiency)
    hours = prices.index.hour
    
    # Load 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1h RSI(14) for momentum
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 8-20 UTC
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI > 50 (bullish momentum) + price above 4h EMA50 + volume spike
            if rsi[i] > 50 and close[i] > ema50_4h_aligned[i] and vol_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: RSI < 50 (bearish momentum) + price below 4h EMA50 + volume spike
            elif rsi[i] < 50 and close[i] < ema50_4h_aligned[i] and vol_spike[i]:
                signals[i] = -0.20
                position = -1
        else:
            # Exit: RSI crosses 50 (momentum shift) OR volume drops
            if position == 1:
                if rsi[i] < 50 or not vol_spike[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if rsi[i] > 50 or not vol_spike[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_RSI_Momentum_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0