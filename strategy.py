# 25
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Ichimoku Cloud Breakout with 1d Trend Filter
# Uses Ichimoku Cloud (Tenkan/Kijun/Senkou) for breakout signals and 1d EMA50 for trend direction.
# Long when price breaks above Kumo cloud in bullish trend (price > 1d EMA50),
# short when price breaks below Kumo cloud in bearish trend (price < 1d EMA50).
# Volume confirmation and ADX filter to avoid false breakouts. Designed for low trade frequency.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d EMA50 for trend filter (higher timeframe)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Ichimoku Cloud components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # ADX for trend strength filter
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - close[:-1]), np.absolute(low[1:] - close[:-1]))
    
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: current > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    for i in range(52, n):  # Start after Ichimoku calculation window
        # Skip if any required data is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or 
            np.isnan(senkou_b[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            continue
        
        # Cloud top and bottom
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])
        
        # Long conditions: price breaks above cloud in bullish trend
        if (adx[i] > 20 and 
            close[i] > cloud_top and 
            close[i-1] <= cloud_top and  # Breakout confirmation
            close[i] > ema_1d_aligned[i] and  # Above 1d EMA50 (bullish trend)
            volume[i] > 1.5 * vol_ma[i]):  # Volume confirmation
            signals[i] = 0.25
        
        # Short conditions: price breaks below cloud in bearish trend
        elif (adx[i] > 20 and 
              close[i] < cloud_bottom and 
              close[i-1] >= cloud_bottom and  # Breakdown confirmation
              close[i] < ema_1d_aligned[i] and  # Below 1d EMA50 (bearish trend)
              volume[i] > 1.5 * vol_ma[i]):  # Volume confirmation
            signals[i] = -0.25
        
        # Exit conditions: price returns inside cloud or trend weakens
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (close[i] < cloud_top or adx[i] <= 20)) or
               (signals[i-1] == -0.25 and (close[i] > cloud_bottom or adx[i] <= 20)))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_Ichimoku_Cloud_Breakout_1dEMA"
timeframe = "4h"
leverage = 1.0