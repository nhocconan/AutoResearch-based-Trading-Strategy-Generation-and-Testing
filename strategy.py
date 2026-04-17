# The 6h timeframe has proven effective when combining multiple timeframe confirmation with volatility-based entry filters. This strategy uses 1-week Ichimoku cloud for trend direction, 1-day ATR for volatility filtering, and price action relative to the 6-day EMA for entry timing. The Ichimoku cloud provides robust trend identification that works in both trending and ranging markets, while the ATR filter ensures we only trade during sufficient volatility periods. The EMA provides dynamic support/resistance for entries. This multi-timeframe approach reduces false signals and improves signal quality.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1-week Ichimoku Cloud for Trend Direction ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a, additional_delay_bars=26)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b, additional_delay_bars=26)
    
    # === 1-day ATR for Volatility Filtering ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.abs(high_1d[0] - low_1d[0])  # First period
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14)
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=10, min_periods=10).mean().values
    
    # Align ATR to 6h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    
    # === 6-day EMA for Dynamic Support/Resistance ===
    ema_6 = pd.Series(close).ewm(span=6, adjust=False, min_periods=6).mean().values
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or 
            np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(atr_ma_1d_aligned[i]) or
            np.isnan(ema_6[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Determine Ichimoku trend: price above/below cloud
        cloud_top = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Volatility filter: only trade when ATR is above its moving average
        volatility_filter = atr_1d_aligned[i] > atr_ma_1d_aligned[i]
        
        # Entry conditions
        if position == 0:
            # Long: price above cloud, above EMA6, and volatility sufficient
            if price_above_cloud and close[i] > ema_6[i] and volatility_filter:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price below cloud, below EMA6, and volatility sufficient
            elif price_below_cloud and close[i] < ema_6[i] and volatility_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit conditions: reverse signal or volatility drops
        elif position == 1:
            # Exit long: price drops below cloud or below EMA6 or volatility drops
            if not (price_above_cloud and close[i] > ema_6[i] and volatility_filter):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above cloud or above EMA6 or volatility drops
            if not (price_below_cloud and close[i] < ema_6[i] and volatility_filter):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1w_Ichimoku_Cloud_Trend_6EMA_VolatilityFilter_v1"
timeframe = "6h"
leverage = 1.0