#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_50_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_50)
    
    # Load daily data for ATR-based volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-day ATR for regime classification
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]
    atr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 50-day SMA of ATR for regime threshold
    atr_ma50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    
    # Align ATR regime indicators to 1h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_ma50_aligned = align_htf_to_ltf(prices, df_1d, atr_ma50)
    
    # Determine volatility regime: HIGH_VOL when ATR > 1.5 * ATR_MA50
    high_vol_regime = atr_14_aligned > (1.5 * atr_ma50_aligned)
    
    # Calculate 1h RSI(14) for mean reversion signals
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter (20-period MA)
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 2.0 * vol_ma20
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_4h_50_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma20[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(atr_ma50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Mean reversion logic based on volatility regime
            if high_vol_regime[i]:
                # High volatility: momentum follows trend
                if rsi[i] > 50 and vol_surge[i] and close[i] > ema_4h_50_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                elif rsi[i] < 50 and vol_surge[i] and close[i] < ema_4h_50_aligned[i]:
                    signals[i] = -0.20
                    position = -1
            else:
                # Low volatility: mean reversion at extremes
                if rsi[i] < 30 and vol_surge[i] and close[i] > ema_4h_50_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                elif rsi[i] > 70 and vol_surge[i] and close[i] < ema_4h_50_aligned[i]:
                    signals[i] = -0.20
                    position = -1
        else:
            # Exit conditions
            if position == 1:
                if rsi[i] > 70 or not in_session[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if rsi[i] < 30 or not in_session[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_RSI_VolRegime_MeanRev_Mom_4hEMA50_VolSurge"
timeframe = "1h"
leverage = 1.0