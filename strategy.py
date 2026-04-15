#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA(34) trend filter and volume confirmation.
# Works in bull markets via breakout continuation; in bear markets via short breakdowns below trend.
# Uses discrete position sizing (0.25) to limit fee churn and drawdown.
# Target: 20-50 trades/year per symbol (<200 total over 4 years) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA(34) for trend filter
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 12h ATR(14) for volatility filter
    tr1 = df_12h['high'] - df_12h['low']
    tr2 = np.abs(df_12h['high'] - np.concatenate([[df_12h['close'].iloc[0]], df_12h['close'].iloc[:-1]]))
    tr3 = np.abs(df_12h['low'] - np.concatenate([[df_12h['close'].iloc[0]], df_12h['close'].iloc[:-1]]))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_14_12h)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(atr_14_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when 12h ATR is elevated (> 0.4% of price)
        vol_filter = atr_14_12h_aligned[i] > 0.004 * close[i]
        
        # Calculate 4h Donchian(20) channels using available data up to i
        start_idx = max(0, i - 19)
        donchian_high = np.max(high[start_idx:i+1])
        donchian_low = np.min(low[start_idx:i+1])
        
        # Long conditions:
        # 1. Price above 12h EMA34 (bullish bias)
        # 2. Price breaks above 4h Donchian(20) high
        # 3. Volume confirmation: current volume > 20-period average
        # 4. Volatility filter
        if (close[i] > ema_34_12h_aligned[i] and
            close[i] > donchian_high and
            vol_filter and
            volume[i] > np.mean(np.maximum(volume[start_idx:i], 1e-9))):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below 12h EMA34 (bearish bias)
        # 2. Price breaks below 4h Donchian(20) low
        # 3. Volume confirmation
        # 4. Volatility filter
        elif (close[i] < ema_34_12h_aligned[i] and
              close[i] < donchian_low and
              vol_filter and
              volume[i] > np.mean(np.maximum(volume[start_idx:i], 1e-9))):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_EMA34_12h_VolFilter_v1"
timeframe = "4h"
leverage = 1.0