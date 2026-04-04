#!/usr/bin/env python3
"""
exp_6460_4h_donchian20_1d_ema_vol_v1
Hypothesis: 4h Donchian(20) breakout with 1d EMA(50) trend filter and volume confirmation.
Works in bull/bear because: Donchian captures breakouts, 1d EMA filters counter-trend trades,
volume confirmation avoids false breakouts. Discrete sizing (0.25) minimizes fee churn.
Target: 75-200 trades over 4 years.
"""
from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6460_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if df_1d is None or len(df_1d) < 60:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_1d = close_1d.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 4h Donchian(20)
    high_s = prices['high']
    low_s = prices['low']
    donch_high = high_s.rolling(window=20, min_periods=20).max()
    donch_low = low_s.rolling(window=20, min_periods=20).min()
    
    # Volume confirmation: 4h volume > 1.5 * 20-period average
    vol_s = prices['volume']
    vol_ma = vol_s.rolling(window=20, min_periods=20).mean()
    vol_ratio = vol_s / vol_ma
    
    # ATR(14) for stoploss
    tr1 = prices['high'] - prices['low']
    tr2 = abs(prices['high'] - prices['close'].shift(1))
    tr3 = abs(prices['low'] - prices['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if indicators not ready
        if pd.isna(donch_high.iloc[i]) or pd.isna(donch_low.iloc[i]) or \
           pd.isna(ema_1d_aligned[i]) or pd.isna(vol_ratio.iloc[i]) or pd.isna(atr.iloc[i]):
            continue
        
        close = prices['close'].iloc[i]
        
        # Stoploss check
        if position == 1 and close < entry_price - 2.0 * atr.iloc[i]:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and close > entry_price + 2.0 * atr.iloc[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic
        if position == 0:
            # Long: price breaks above Donchian high, above 1d EMA, volume spike
            if (close > donch_high.iloc[i] and 
                close > ema_1d_aligned[i] and 
                vol_ratio.iloc[i] > 1.5):
                signals[i] = 0.25
                position = 1
                entry_price = close
            # Short: price breaks below Donchian low, below 1d EMA, volume spike
            elif (close < donch_low.iloc[i] and 
                  close < ema_1d_aligned[i] and 
                  vol_ratio.iloc[i] > 1.5):
                signals[i] = -0.25
                position = -1
                entry_price = close
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else -0.25
    
    return signals