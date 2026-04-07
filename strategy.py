#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h Stochastic RSI + 4h Trend + 1d Volume Confirmation
# Hypothesis: In strong trends (4h EMA20 > EMA50), buy pullbacks on 1h when StochRSI < 0.2 (oversold),
# sell rallies when StochRSI > 0.8 (overbought), confirmed by 1d volume > 1.5x average.
# Works in bull/bear: trend filter ensures we trade with higher timeframe momentum,
# while StochRSI captures mean-reversion within the trend. Volume filter avoids low-liquidity traps.
# Target: 15-35 trades/year (~60-140 over 4 years) to minimize fee drag.
name = "1h_stochrsi_4h_trend_1d_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h EMA20 and EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1d volume average for filter
    volume_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # 1h Stochastic RSI (14,14,3,3)
    # RSI calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Stochastic RSI
    rsi_min = pd.Series(rsi).rolling(window=14, min_periods=14).min().values
    rsi_max = pd.Series(rsi).rolling(window=14, min_periods=14).max().values
    stoch_rsi = np.where((rsi_max - rsi_min) != 0, (rsi - rsi_min) / (rsi_max - rsi_min), 0)
    # Smooth with K=3, D=3
    stoch_rsi_k = pd.Series(stoch_rsi).rolling(window=3, min_periods=3).mean().values
    stoch_rsi_d = pd.Series(stoch_rsi_k).rolling(window=3, min_periods=3).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or
            np.isnan(vol_avg_1d_aligned[i]) or np.isnan(stoch_rsi_d[i])):
            signals[i] = 0.0
            continue
        
        # Check session
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: trend turns bearish or StochRSI overbought
            if ema20_4h_aligned[i] < ema50_4h_aligned[i] or stoch_rsi_d[i] > 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long position
        elif position == -1:  # Short position
            # Exit: trend turns bullish or StochRSI oversold
            if ema20_4h_aligned[i] > ema50_4h_aligned[i] or stoch_rsi_d[i] < 0.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation and in-session
            if volume[i] > (vol_avg_1d_aligned[i] * 1.5):
                # Long: uptrend (EMA20 > EMA50) + StochRSI oversold
                if ema20_4h_aligned[i] > ema50_4h_aligned[i] and stoch_rsi_d[i] < 0.2:
                    position = 1
                    signals[i] = 0.20
                # Short: downtrend (EMA20 < EMA50) + StochRSI overbought
                elif ema20_4h_aligned[i] < ema50_4h_aligned[i] and stoch_rsi_d[i] > 0.8:
                    position = -1
                    signals[i] = -0.20
    
    return signals