#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation
    # Tenkan/Kijun cross signals momentum; Cloud (Senkou Span) provides dynamic support/resistance
    # Filtering by 1d EMA50 ensures trades align with higher timeframe trend
    # Volume confirmation adds conviction, reducing false signals
    # Designed for 6h timeframe to target 12-37 trades/year with disciplined risk
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 6h data for Ichimoku calculation
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Ichimoku components (9, 26, 52)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_6h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_6h).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h timeframe (properly shifted for look-ahead avoidance)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_6h, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_6h, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_b)
    
    # Load 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: 20-period volume average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume > 1.5 * vol_ma20  # Require 1.5x average volume
    
    # Session filter: 08-20 UTC (avoid low-liquidity periods)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after Ichimoku warmup (52 periods)
        # Skip if data not ready or outside session
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud top and bottom
        cloud_top = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        if position == 0:
            # Long: TK cross bullish + price above cloud + volume + uptrend (price > 1d EMA50)
            if (tenkan_sen_aligned[i] > kijun_sen_aligned[i] and  # TK cross bullish
                close[i] > cloud_top and  # Price above cloud
                vol_surge[i] and  # Volume confirmation
                close[i] > ema50_1d_aligned[i]):  # Uptrend filter
                signals[i] = 0.25
                position = 1
            # Short: TK cross bearish + price below cloud + volume + downtrend (price < 1d EMA50)
            elif (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and  # TK cross bearish
                  close[i] < cloud_bottom and  # Price below cloud
                  vol_surge[i] and  # Volume confirmation
                  close[i] < ema50_1d_aligned[i]):  # Downtrend filter
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: TK cross reverse OR price returns to cloud
            if position == 1:
                # Exit long: TK cross bearish OR price drops below cloud
                if (tenkan_sen_aligned[i] < kijun_sen_aligned[i] or  # TK cross bearish
                    close[i] < cloud_top):  # Price below cloud top
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: TK cross bullish OR price rises above cloud
                if (tenkan_sen_aligned[i] > kijun_sen_aligned[i] or  # TK cross bullish
                    close[i] > cloud_bottom):  # Price above cloud bottom
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_Cloud_Filter_1dEMA50_Volume_v1"
timeframe = "6h"
leverage = 1.0