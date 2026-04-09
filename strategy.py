#!/usr/bin/env python3
# 12h_hma_rsi_1w_volume_v1
# Hypothesis: 12h strategy using HMA trend filter and RSI mean reversion with weekly volume confirmation.
# Enters long when HMA(21) is rising and RSI(14) < 30 with volume spike, short when HMA(21) is falling and RSI(14) > 70 with volume spike.
# Uses discrete sizing (±0.25) to minimize fee churn. Designed for low trade frequency (target: 50-150 total trades over 4 years).
# Works in bull/bear by using HMA for trend direction and RSI for mean reversion entries.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_hma_rsi_1w_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h EMA for HMA calculation
    close_s = pd.Series(close)
    ema_12 = close_s.ewm(span=12, min_periods=12, adjust=False).mean().values
    ema_24 = close_s.ewm(span=24, min_periods=24, adjust=False).mean().values
    raw_hma = 2 * ema_12 - ema_24
    hma = pd.Series(raw_hma).ewm(span=int(np.sqrt(12)) + 1, min_periods=int(np.sqrt(12)) + 1, adjust=False).mean().values
    
    # 1w HTF data for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    volume_1w = df_1w['volume'].values
    vol_ma_20 = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_spike_1w = volume_1w > (vol_ma_20 * 2.0)
    vol_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_spike_1w)
    
    # RSI(14) calculation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(hma[i]) or np.isnan(rsi[i]) or np.isnan(vol_spike_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: HMA turns down OR RSI > 50
            if hma[i] < hma[i-1] or rsi[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: HMA turns up OR RSI < 50
            if hma[i] > hma[i-1] or rsi[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: HMA rising AND RSI < 30 with volume spike
            if (hma[i] > hma[i-1]) and \
               (rsi[i] < 30) and \
               (vol_spike_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: HMA falling AND RSI > 70 with volume spike
            elif (hma[i] < hma[i-1]) and \
                 (rsi[i] > 70) and \
                 (vol_spike_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals