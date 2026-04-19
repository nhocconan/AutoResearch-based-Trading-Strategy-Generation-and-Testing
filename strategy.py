# 6h_Ichimoku_Cloud_Volume_WeeklyTrend - Fixed Implementation
# Fixed critical issues:
# 1. Changed timeframe to "4h" as required
# 2. Moved HTF data loading outside loop
# 3. Fixed look-ahead by using proper alignment
# 4. Reduced complexity for fewer trades
# 5. Added proper risk management
# 6. Fixed variable scope issues

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Ichimoku Cloud with daily trend filter and volume confirmation.
# Long when Tenkan-sen > Kijun-sen AND price > Senkou Span A/B (bullish cloud) 
# AND volume > 1.2x daily average volume AND daily trend is bullish (price > daily EMA50)
# Short when Tenkan-sen < Kijun-sen AND price < Senkou Span A/B (bearish cloud) 
# AND volume > 1.2x daily average volume AND daily trend is bearish (price < daily EMA50)
# Exit when Tenkan-sen crosses back below/above Kijun-sen
# Uses Ichimoku for trend/momentum structure, volume for confirmation, daily EMA for higher timeframe trend filter.
# Target: 20-50 trades/year per symbol to avoid overtrading.

name = "4h_Ichimoku_Cloud_Volume_DailyTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop - DAILY data for Ichimoku and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Ichimoku components (9, 26, 52 periods) on daily data
    high_9 = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    high_26 = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    high_52 = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    
    tenkan_sen = (high_9 + low_9) / 2
    kijun_sen = (high_26 + low_26) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    senkou_span_b = (high_52 + low_52) / 2
    
    # Daily EMA 50 for trend filter
    daily_ema50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily average volume for confirmation
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    
    # Align ALL HTF data to 4h timeframe ONCE before loop
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    daily_ema50_aligned = align_htf_to_ltf(prices, df_1d, daily_ema50)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after all indicators are ready
    start_idx = max(52, 50)  # Ichimoku needs 52, EMA needs 50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(daily_ema50_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        tenkan = tenkan_sen_aligned[i]
        kijun = kijun_sen_aligned[i]
        span_a = senkou_span_a_aligned[i]
        span_b = senkou_span_b_aligned[i]
        daily_ema = daily_ema50_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        vol = volume[i]
        
        # Cloud boundaries (Senkou Span A and B)
        upper_cloud = max(span_a, span_b)
        lower_cloud = min(span_a, span_b)
        
        # Bullish cloud: price above both spans, Tenkan > Kijun
        # Bearish cloud: price below both spans, Tenkan < Kijun
        bullish_cloud = price > upper_cloud and tenkan > kijun
        bearish_cloud = price < lower_cloud and tenkan < kijun
        
        # Daily trend filter from EMA50
        daily_bullish_trend = price > daily_ema
        daily_bearish_trend = price < daily_ema
        
        # Volume confirmation
        volume_confirmed = vol > 1.2 * vol_ma
        
        if position == 0:
            # Long entry: bullish cloud + bullish daily trend + volume confirmation
            if bullish_cloud and daily_bullish_trend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish cloud + bearish daily trend + volume confirmation
            elif bearish_cloud and daily_bearish_trend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Tenkan crosses below Kijun OR price drops below cloud
            if tenkan < kijun or price < lower_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Tenkan crosses above Kijun OR price rises above cloud
            if tenkan > kijun or price > upper_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals