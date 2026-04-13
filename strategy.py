#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for ATR and CCI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Daily ATR for volatility filter (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily CCI (20-period)
    tp_1d = (high_1d + low_1d + close_1d) / 3
    sma_tp_20 = pd.Series(tp_1d).rolling(window=20, min_periods=20).mean()
    mad = pd.Series(tp_1d).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    cci_20_1d = (tp_1d - sma_tp_20) / (0.015 * mad)
    cci_20_1d = cci_20_1d.values
    
    # Weekly EMA for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all data to 1h timeframe
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    cci_20_1d_aligned = align_htf_to_ltf(prices, df_1d, cci_20_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1h ATR for entry trigger
    tr1h = high - low
    tr2h = np.abs(high - np.roll(close, 1))
    tr3h = np.abs(low - np.roll(close, 1))
    tr_h = np.maximum(tr1h, np.maximum(tr2h, tr3h))
    tr_h[0] = tr1h[0]
    atr_14_1h = pd.Series(tr_h).rolling(window=14, min_periods=14).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.20  # 20% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(cci_20_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr_14_1h[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when daily ATR is above its 50-period average
        atr_ma_50_1d = pd.Series(atr_14_1d_aligned[:i+1]).rolling(window=50, min_periods=50).mean()
        if len(atr_ma_50_1d) < 50 or np.isnan(atr_ma_50_1d.iloc[-1]):
            vol_filter = False
        else:
            vol_filter = atr_14_1d_aligned[i] > atr_ma_50_1d.iloc[-1]
        
        # Trend filter: EMA50 on weekly
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # CCI conditions for mean reversion
        cci_oversold = cci_20_1d_aligned[i] < -100
        cci_overbought = cci_20_1d_aligned[i] > 100
        
        # Entry conditions: volatility + trend + CCI extreme
        if position == 0:
            if vol_filter and uptrend and cci_oversold:
                # Long in uptrend when oversold
                position = 1
                signals[i] = position_size
            elif vol_filter and downtrend and cci_overbought:
                # Short in downtrend when overbought
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when CCI returns to neutral or trend changes
            if cci_20_1d_aligned[i] >= 0 or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when CCI returns to neutral or trend changes
            if cci_20_1d_aligned[i] <= 0 or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_1d1w_ATR_CCI_MeanReversion_TrendFilter_v1"
timeframe = "1h"
leverage = 1.0