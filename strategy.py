#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h VWAP pullback with 4h trend filter and daily volume confirmation
# Hypothesis: In uptrends, price pulls back to VWAP offers long opportunities; in downtrends, offers short opportunities.
# Uses 4h EMA for trend direction, daily volume for confirmation, 1h VWAP for entry timing.
# Works in bull via trend-following pullbacks, in bear via mean reversion to VWAP in downtrends.
# Target: 15-37 trades/year to minimize fee drag.
name = "1h_vwap_pullback_4h1d_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 8-20 UTC (avoid low-volume Asian session)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend direction
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily 20-period volume moving average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 1h VWAP (volume-weighted average price)
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = vwap_numerator / vwap_denominator
    
    # Calculate ATR(14) for dynamic thresholds
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    for i in range(50, n):  # Start after warmup period
        # Skip if required data not available or outside session
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(vwap[i]) or np.isnan(atr[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 4h EMA50
        uptrend = close[i] > ema_4h_aligned[i]
        downtrend = close[i] < ema_4h_aligned[i]
        
        # Volume confirmation: current volume > daily average volume
        vol_confirm = volume[i] > vol_ma_1d_aligned[i]
        
        # VWAP deviation threshold: 0.5 * ATR
        vwap_dev = (close[i] - vwap[i]) / atr[i]
        
        if uptrend and vol_confirm:
            # Long when price pulls back to VWAP in uptrend
            if -0.5 <= vwap_dev <= 0.5:  # Near VWAP
                signals[i] = 0.20
            elif vwap_dev < -0.5:  # Oversold - mean reversion long
                signals[i] = 0.20
            else:
                signals[i] = 0.0
        elif downtrend and vol_confirm:
            # Short when price rallies to VWAP in downtrend
            if -0.5 <= vwap_dev <= 0.5:  # Near VWAP
                signals[i] = -0.20
            elif vwap_dev > 0.5:  # Overbought - mean reversion short
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals