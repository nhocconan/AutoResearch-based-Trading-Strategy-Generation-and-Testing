#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation
# Long when price above cloud (Senkou Span A/B) + TK cross bullish + 1d close > 1d EMA50 + volume > 1.5x 60-period 6h volume average
# Short when price below cloud + TK cross bearish + 1d close < 1d EMA50 + volume > 1.5x 60-period 6h volume average
# Exit when price crosses opposite Kumo (cloud) boundary or TK cross reverses
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Ichimoku provides clear trend definition and support/resistance; higher timeframe filter avoids counter-trend trades
# Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_ichimoku_1d_trend_vol_v1"
timeframe = "6h"
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
    
    # 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
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
    
    # Chikou Span (Lagging Span): current close plotted 26 periods back
    # Not used for signals but needed for cloud calculation alignment
    
    # The cloud is between Senkou Span A and Senkou Span B
    # For plotting, Senkou spans are shifted forward 26 periods
    # For signal generation, we use current Senkou spans (no shift) as support/resistance
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume filter: volume > 1.5x 60-period 6h volume average
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=60, min_periods=60).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(52, n):  # Need 52 periods for Senkou B
        # Skip if required data not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A and B)
        upper_cloud = np.maximum(senkou_a[i], senkou_b[i])
        lower_cloud = np.minimum(senkou_a[i], senkou_b[i])
        
        # TK Cross
        tk_bullish = tenkan[i] > kijun[i]
        tk_bearish = tenkan[i] < kijun[i]
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price falls below cloud OR TK cross turns bearish
            elif close[i] < lower_cloud or not tk_bullish:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price rises above cloud OR TK cross turns bullish
            elif close[i] > upper_cloud or tk_bullish:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: price relative to cloud + TK cross + 1d trend + volume
            # Volume filter
            volume_filter = volume[i] > 1.5 * volume_ma[i]
            # 1d trend filter
            trend_filter_long = close[i] > ema50_1d_aligned[i]
            trend_filter_short = close[i] < ema50_1d_aligned[i]
            
            # Long: price above cloud + TK bullish + 1d uptrend + volume
            if (close[i] > upper_cloud and tk_bullish and 
                trend_filter_long and volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price below cloud + TK bearish + 1d downtrend + volume
            elif (close[i] < lower_cloud and tk_bearish and 
                  trend_filter_short and volume_filter):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals