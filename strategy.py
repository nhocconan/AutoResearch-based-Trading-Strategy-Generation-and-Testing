#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h RSI mean reversion with 4h/1d trend filter and volume confirmation
    # Uses 4h EMA50 and 1d EMA200 for multi-timeframe trend alignment (bullish/bearish regime)
    # 1h RSI(14) < 30 for long entry, > 70 for short entry in alignment with higher TF trend
    # Volume confirmation (>1.5x 20-period average) filters weak breakouts
    # Session filter (08-20 UTC) reduces low-liquidity noise
    # Discrete position sizing (0.20) to minimize fee churn
    # Target: 15-30 trades/year (60-120 total over 4 years) to stay within fee drag limits
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d data for trend filters
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    close_1d = df_1d['close'].values
    volume_1h = prices['volume'].values
    
    # 4h EMA50 for medium-term trend
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d EMA200 for long-term trend
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # 1h RSI(14) for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 1h volume confirmation (>1.5x 20-period average)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # Align HTF indicators to LTF (1h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Trend alignment conditions
        bullish_regime = close[i] > ema50_4h_aligned[i] and close[i] > ema200_1d_aligned[i]
        bearish_regime = close[i] < ema50_4h_aligned[i] and close[i] < ema200_1d_aligned[i]
        
        # RSI mean reversion entries
        long_entry = rsi[i] < 30 and bullish_regime and volume_confirm[i]
        short_entry = rsi[i] > 70 and bearish_regime and volume_confirm[i]
        
        # Exit conditions: RSI returns to neutral zone (40-60) or trend breaks
        long_exit = rsi[i] > 40 or not bullish_regime
        short_exit = rsi[i] < 60 or not bearish_regime
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_rsi_meanreversion_ema50_ema200_volume_v1"
timeframe = "1h"
leverage = 1.0