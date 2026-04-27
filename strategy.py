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
    
    # Get 1d data for trend, volatility, and price channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily EMA(34) for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Daily ATR(14) for volatility
    tr1_d = df_1d['high'].values - df_1d['low'].values
    tr2_d = np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1))
    tr3_d = np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
    tr_d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    tr_d[0] = tr1_d[0]
    atr_1d_raw = pd.Series(tr_d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_raw)
    
    # 1h ATR(14) for volatility filter
    tr1_h = high - low
    tr2_h = np.abs(high - np.roll(close, 1))
    tr3_h = np.abs(low - np.roll(close, 1))
    tr_h = np.maximum(tr1_h, np.maximum(tr2_h, tr3_h))
    tr_h[0] = tr1_h[0]
    atr_1h = pd.Series(tr_h).rolling(window=14, min_periods=14).mean().values
    
    # 1h RSI(14) for momentum confirmation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 1h Williams %R for mean reversion signals
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # Position size: 20% of capital
    
    # Warmup
    start_idx = max(34, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(atr_1h[i]) or 
            i >= len(atr_1d_aligned) or np.isnan(atr_1d_aligned[i]) or
            np.isnan(rsi[i]) or np.isnan(williams_r[i])):
            signals[i] = 0.0
            continue
        
        ema_trend = ema34_1d_aligned[i]
        atr_1h_val = atr_1h[i]
        atr_1d_val = atr_1d_aligned[i]
        rsi_val = rsi[i]
        williams_r_val = williams_r[i]
        
        # Volatility filter: 1h ATR > 0.3 * daily ATR (avoid low volatility)
        vol_filter = atr_1h_val > (atr_1d_val * 0.3)
        
        # RSI filter: avoid extremes
        rsi_filter = (rsi_val > 25) & (rsi_val < 75)
        
        # Williams %R: oversold/overbought levels
        wr_oversold = williams_r_val < -80
        wr_overbought = williams_r_val > -20
        
        if position == 0:
            # Long: price above EMA with volatility, RSI filter, and Williams %R oversold
            if close[i] > ema_trend and vol_filter and rsi_filter and wr_oversold:
                signals[i] = size
                position = 1
            # Short: price below EMA with volatility, RSI filter, and Williams %R overbought
            elif close[i] < ema_trend and vol_filter and rsi_filter and wr_overbought:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below EMA or Williams %R overbought
            if close[i] < ema_trend or williams_r_val > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above EMA or Williams %R oversold
            if close[i] > ema_trend or williams_r_val < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_EMA34_Trend_WilliamsR_VolumeFilter_v1"
timeframe = "1h"
leverage = 1.0