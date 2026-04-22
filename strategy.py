#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(2) mean reversion with 4h trend filter and 1d volatility filter
# RSI(2) < 10 for long, > 90 for short in trending markets (4h EMA50 direction)
# Volatility filter: avoid trading when 1d ATR(10) is in lowest 20% (low volatility environments)
# Time session: 08-20 UTC to avoid low-volume Asian session
# Designed for 1h timeframe targeting 15-35 trades/year per symbol
# Works in bull markets by buying dips in uptrends, in bear markets by selling rallies in downtrends

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for trend filter (ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data for volatility filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ATR(10) for volatility regime filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_10_1d = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr_20_1d = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    # Avoid low volatility: require ATR(10) > 20th percentile of ATR(20)
    vol_filter = atr_10_1d > (atr_20_1d * 0.8)
    vol_filter_aligned = align_htf_to_ltf(prices, df_1d, vol_filter)
    
    # Calculate RSI(2) on 1h data
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ema = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    loss_ema = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = gain_ema / (loss_ema + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI(2) < 10 + 4h uptrend + volatility filter
            if (rsi[i] < 10 and 
                close[i] > ema_50_4h_aligned[i] and 
                vol_filter_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: RSI(2) > 90 + 4h downtrend + volatility filter
            elif (rsi[i] > 90 and 
                  close[i] < ema_50_4h_aligned[i] and 
                  vol_filter_aligned[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions: RSI returns to neutral zone or trend reversal
            if position == 1:
                # Exit on RSI >= 50 or trend reversal
                if (rsi[i] >= 50 or 
                    close[i] < ema_50_4h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                # Exit on RSI <= 50 or trend reversal
                if (rsi[i] <= 50 or 
                    close[i] > ema_50_4h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_RSI2_4hEMA50_Trend_1dVolFilter_Session"
timeframe = "1h"
leverage = 1.0