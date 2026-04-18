#!/usr/bin/env python3
"""
1h_4HTrend_RSI20_80_1DVolume
Hypothesis: In both bull and bear markets, 4h EMA50 trend provides directional bias.
On 1h timeframe, we wait for RSI extremes (20 oversold, 80 overbought) to enter
in direction of 4h trend, confirmed by 1d volume spike. Designed for 15-30 trades/year
on 1h timeframe with low frequency to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA trend (once before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close']
    
    # 4h EMA50 trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume']
    
    # 1d volume spike: 2x 20-period average
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # 1h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(volume_spike_1d_aligned[i]) or
            np.isnan(rsi_values[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_trend = ema_50_4h_aligned[i]
        vol_spike = volume_spike_1d_aligned[i]
        rsi_val = rsi_values[i]
        
        if position == 0:
            # Long: 4h uptrend + RSI oversold + 1d volume spike
            if price > ema_trend and rsi_val < 20 and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend + RSI overbought + 1d volume spike
            elif price < ema_trend and rsi_val > 80 and vol_spike:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.20
            # Exit: RSI returns to neutral or trend breaks
            if rsi_val >= 50 or price < ema_trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.20
            # Exit: RSI returns to neutral or trend breaks
            if rsi_val <= 50 or price > ema_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_4HTrend_RSI20_80_1DVolume"
timeframe = "1h"
leverage = 1.0