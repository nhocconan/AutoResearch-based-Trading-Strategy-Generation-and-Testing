#!/usr/bin/env python3
"""
1h RSI Mean Reversion with 4h Trend Filter and Volume Spike
Hypothesis: In 1h timeframe, RSI mean reversion works best when aligned with 4h trend direction.
During strong trends, pullbacks to RSI extremes offer high-probability entries. Volume spike
confirms participation at turning points. This reduces false signals in ranging markets.
Designed for 15-30 trades/year to minimize fee drag on 1h chart.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean()
    loss_ma = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean()
    rs = gain_ma / loss_ma
    rsi = 100 - (100 / (1 + rs))
    
    # 4h EMA(34) for trend
    df_4h = get_htf_data(prices, '4h')
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Volume spike: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for RSI and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        ema_trend = ema_4h_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: RSI oversold in uptrend with volume spike
            if rsi_val < 30 and price > ema_trend and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought in downtrend with volume spike
            elif rsi_val > 70 and price < ema_trend and vol_spike:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: RSI overbought or trend reversal
            if rsi_val > 70 or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: RSI oversold or trend reversal
            if rsi_val < 30 or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI_MeanReversion_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0