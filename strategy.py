#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h trend filter and 1d volatility filter.
# Uses 1h RSI(14) for momentum entry/exit, 4h EMA(50) for trend direction,
# and 1d ATR(14) normalized by price for volatility regime filtering.
# Only trades during 08-20 UTC to avoid low-liquidity hours.
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
# Designed to work in both bull (momentum continuation) and bear (mean reversion in low volatility).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1h RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    for i in range(1, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h EMA(50) for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema50_4h = np.zeros(len(close_4h))
    ema_multiplier = 2 / (50 + 1)
    ema50_4h[0] = close_4h[0]
    for i in range(1, len(close_4h)):
        ema50_4h[i] = (close_4h[i] - ema50_4h[i-1]) * ema_multiplier + ema50_4h[i-1]
    
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1d ATR(14) normalized by price for volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.zeros(len(close_1d))
    tr1[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(close_1d)):
        tr1[i] = max(high_1d[i] - low_1d[i], 
                     abs(high_1d[i] - close_1d[i-1]),
                     abs(low_1d[i] - close_1d[i-1]))
    
    atr14 = np.zeros(len(close_1d))
    atr14[0] = tr1[0]
    for i in range(1, len(atr14)):
        atr14[i] = (atr14[i-1] * 13 + tr1[i]) / 14
    
    # Normalize ATR by price to get volatility regime
    atr_norm = atr14 / close_1d
    atr_norm_aligned = align_htf_to_ltf(prices, df_1d, atr_norm)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.20  # 20% position size
    
    for i in range(50, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is not ready
        if (np.isnan(rsi[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(atr_norm_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        ema_trend = ema50_4h_aligned[i]
        vol_regime = atr_norm_aligned[i]
        
        # Volatility filter: only trade when volatility is normalized (not too high)
        # Avoid trading during extreme volatility spikes
        vol_filter = vol_regime < 0.05  # 5% daily ATR as threshold
        
        if position == 0:
            # Long: RSI > 55 (bullish momentum) + above 4h EMA50 + vol filter
            if (rsi_val > 55 and 
                price > ema_trend and
                vol_filter):
                position = 1
                signals[i] = position_size
            # Short: RSI < 45 (bearish momentum) + below 4h EMA50 + vol filter
            elif (rsi_val < 45 and 
                  price < ema_trend and
                  vol_filter):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI < 40 (momentum fade) or price below 4h EMA
            if (rsi_val < 40 or
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI > 60 (momentum fade) or price above 4h EMA
            if (rsi_val > 60 or
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_1d_RSI_EMA_Volatility_Filter"
timeframe = "1h"
leverage = 1.0