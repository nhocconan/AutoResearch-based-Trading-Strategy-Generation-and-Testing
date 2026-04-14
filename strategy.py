# This strategy uses 1-hour timeframe with 4-hour and 1-day higher timeframe filters
# Hypothesis: Combining 4-hour trend direction with 1-day volatility regime and 1-hour momentum
# creates high-probability entries with low turnover. In bull markets, we take long positions
# when 4hr trend is up and 1hr momentum is strong. In bear markets, we take short positions
# when 4hr trend is down and 1hr momentum is weak. The 1-day volatility filter avoids
# choppy markets where whipsaws occur. Designed for ~20-40 trades/year to minimize fee drag.
# Works in both bull and bear via symmetric long/short logic with trend/momentum/volatility filters.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4-hour data for trend (once before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Load 1-day data for volatility regime (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4-hour EMA20 for trend direction
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1-day ATR for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d, additional_delay_bars=0)
    
    # 1-hour RSI for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20
    
    for i in range(50, n):
        if not in_session[i]:
            continue
            
        # Get aligned values for current bar
        trend = ema_4h_aligned[i]
        vol_regime = atr_1d_aligned[i]
        momentum = rsi[i]
        
        # Skip if any value is NaN
        if np.isnan(trend) or np.isnan(vol_regime) or np.isnan(momentum):
            continue
            
        # Calculate 1-hour EMA20 for entry timing
        if i >= 20:
            ema_1h = pd.Series(close[:i+1]).ewm(span=20, adjust=False, min_periods=20).mean().iloc[-1]
        else:
            continue
            
        if position == 0:
            # Long: 4hr uptrend, low volatility, bullish momentum
            if (close[i] > ema_1h and 
                close[i] > trend and 
                vol_regime < np.nanpercentile(atr_1d_aligned[:i+1], 50) and  # below median volatility
                momentum > 50):
                position = 1
                signals[i] = position_size
            # Short: 4hr downtrend, low volatility, bearish momentum
            elif (close[i] < ema_1h and 
                  close[i] < trend and 
                  vol_regime < np.nanpercentile(atr_1d_aligned[:i+1], 50) and  # below median volatility
                  momentum < 50):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit long: trend breaks down or momentum fades
            if close[i] < trend or momentum < 40:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit short: trend breaks up or momentum fades
            if close[i] > trend or momentum > 60:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_Trend_Momentum_Volatility"
timeframe = "1h"
leverage = 1.0