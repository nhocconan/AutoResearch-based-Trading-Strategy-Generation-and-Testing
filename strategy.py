#!/usr/bin/env python3
"""
1h Volume Spike Reversal with 4h Trend and 1d Momentum Filter
Hypothesis: After extreme volume spikes (>3x 20-period avg), price often reverses short-term.
We take mean-reversion trades in direction of 4h trend (avoid counter-trend) and only when 
1d RSI is not overbought/oversold (avoid fading strong trends). Works in bull/bear by 
aligning with higher timeframe momentum. Targets 15-30 trades/year on 1h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_volume_spike_reversal_4h_trend_1d_momentum_v1"
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
    
    # 4h EMA(50) for trend filter
    df_4h = get_htf_data(prices, '4h')
    ema_50_4h = df_4h['close'].ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d RSI(14) for momentum filter (avoid extreme readings)
    df_1d = get_htf_data(prices, '1d')
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_values = rsi_1d.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_values)
    
    # Volume spike detector: >3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 3.0)
    
    # Price change direction for mean reversion: look for exhaustion
    price_change = (close - np.roll(close, 1)) / np.roll(close, 1)
    price_change[0] = 0  # First bar has no previous
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after volume MA warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_spike[i]) or np.isnan(price_change[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes back above VWAP-ish level or volume spike ends
            if (price_change[i] > 0 or not vol_spike[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes back below VWAP-ish level or volume spike ends
            if (price_change[i] < 0 or not vol_spike[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry on volume spike
            # Only trade in direction of 4h trend
            uptrend = close[i] > ema_50_4h_aligned[i]
            downtrend = close[i] < ema_50_4h_aligned[i]
            
            # Avoid extreme 1d RSI readings (overbought/oversold)
            rsi_not_extreme = (rsi_1d_aligned[i] > 20) and (rsi_1d_aligned[i] < 80)
            
            if vol_spike[i] and rsi_not_extreme:
                # Volume spike with negative price change = potential exhaustion for long
                if price_change[i] < -0.005 and uptrend:  # Strong down move in uptrend
                    position = 1
                    signals[i] = 0.20
                # Volume spike with positive price change = potential exhaustion for short
                elif price_change[i] > 0.005 and downtrend:  # Strong up move in downtrend
                    position = -1
                    signals[i] = -0.20
    
    return signals