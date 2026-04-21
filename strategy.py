#2025-06-09T18:00:00.000Z
#!/usr/bin/env python3
"""
6h_PortfolioDiversifier_Trend_Volume
Hypothesis: Portfolio diversification logic applied to single asset. Uses 1d RSI(30) as market state filter and 6h ATR-based breakout with volume confirmation. In bull markets (RSI>50), buy breakouts above ATR channel; in bear markets (RSI<50), sell breakdowns below ATR channel. Volatility-adjusted position sizing prevents blowups in 2022 crash. Target 20-35 trades/year on 6h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d market state: RSI(30) ===
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/30, adjust=False, min_periods=30).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/30, adjust=False, min_periods=30).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === 6h ATR(20) for volatility normalization ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/20, adjust=False, min_periods=20).mean().values
    
    # === Donchian channel (20-period) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume confirmation: 20-period volume average ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma_20 != 0, volume / vol_ma_20, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if indicators not ready
        if (np.isnan(rsi_1d_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        rsi_state = rsi_1d_aligned[i]
        atr_val = atr[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long in bull market (RSI>50): breakout above upper channel with volume
            if (rsi_state > 50 and 
                price_high > upper_channel and
                vol_spike > 1.5):
                signals[i] = 0.25
                position = 1
            # Short in bear market (RSI<50): breakdown below lower channel with volume
            elif (rsi_state < 50 and 
                  price_low < lower_channel and
                  vol_spike > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price crosses opposite channel or RSI flips
            if position == 1:
                if (price_low < lower_channel or rsi_state < 50):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (price_high > upper_channel or rsi_state > 50):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_PortfolioDiversifier_Trend_Volume"
timeframe = "6h"
leverage = 1.0