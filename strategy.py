#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h RSI Pullback with 4h EMA Trend and Volume Spike
# Hypothesis: In strong trends (4h EMA), 1h RSI pullbacks offer high-probability entries.
# Volume spikes confirm institutional interest. Works in bull/bear by following 4h trend.
# Target: 15-37 trades/year (60-150 total over 4 years) using 4h trend + 1h timing.

name = "1h_rsi_pullback_4h_ema_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # 4h EMA(30) for trend filter
    ema_30_4h = pd.Series(close_4h).ewm(span=30, adjust=False).mean().values
    ema_30_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_30_4h)
    
    # 1h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(ema_30_4h_aligned[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: RSI > 70 (overbought) or trend turns bearish
            if rsi_values[i] > 70 or close[i] < ema_30_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short position
            # Exit: RSI < 30 (oversold) or trend turns bullish
            if rsi_values[i] < 30 or close[i] > ema_30_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            if vol_ok:
                # Pullback to uptrend: RSI < 40 in uptrend
                if rsi_values[i] < 40 and close[i] > ema_30_4h_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                # Pullback to downtrend: RSI > 60 in downtrend
                elif rsi_values[i] > 60 and close[i] < ema_30_4h_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals