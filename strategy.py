#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1wTrend
Hypothesis: Ichimoku TK cross with cloud filter and weekly trend filter on 6h timeframe.
- Tenkan-sen (9-period) / Kijun-sen (26-period) cross for entry signals
- Price must be above/below Kumo (cloud) for trend confirmation
- Weekly trend filter: price above/below weekly EMA21 to avoid counter-trend trades
- Works in bull/bear via weekly trend filter and cloud as dynamic support/resistance
- Targets 12-25 trades/year by requiring confluence of TK cross, cloud position, and weekly trend
"""

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
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate Ichimoku components (6h timeframe)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    highest_high_9 = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    lowest_low_9 = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (highest_high_9 + lowest_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    highest_high_26 = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    lowest_low_26 = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (highest_high_26 + lowest_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    highest_high_52 = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    lowest_low_52 = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = (highest_high_52 + lowest_low_52) / 2
    
    # Chikou Span (Lagging Span): not used for signals (would require look-ahead)
    
    # Align Ichimoku components to 6h (already calculated on 6h, no alignment needed)
    # But we need to shift Senkou spans forward by 26 periods (they are plotted 26 periods ahead)
    # For signal generation, we use current Senkou spans (which represent future cloud)
    # Actually, for trading we use the cloud that was plotted 26 periods ago
    senkou_span_a_lagged = np.roll(senkou_span_a, 26)
    senkou_span_b_lagged = np.roll(senkou_span_b, 26)
    # Fill first 26 values with NaN (will be handled by min_periods logic)
    senkou_span_a_lagged[:26] = np.nan
    senkou_span_b_lagged[:26] = np.nan
    
    # Weekly trend filter: EMA21 on weekly timeframe
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema_21_1w = close_1w_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Fixed position size to control trade frequency and drawdown
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 52 for Senkou B, 26 for Kijun, 21 for weekly EMA
    start_idx = max(52, 26, 21) + 26  # +26 for Senkou lag
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen[i]) or
            np.isnan(kijun_sen[i]) or
            np.isnan(senkou_span_a_lagged[i]) or
            np.isnan(senkou_span_b_lagged[i]) or
            np.isnan(ema_21_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        tenkan = tenkan_sen[i]
        kijun = kijun_sen[i]
        senkou_a = senkou_span_a_lagged[i]
        senkou_b = senkou_span_b_lagged[i]
        ema_21_w = ema_21_1w_aligned[i]
        
        # Cloud boundaries (Senkou Span A and B)
        upper_cloud = max(senkou_a, senkou_b)
        lower_cloud = min(senkou_a, senkou_b)
        
        # TK cross signals
        tk_cross_up = tenkan > kijun and tenkan_sen[i-1] <= kijun_sen[i-1]
        tk_cross_down = tenkan < kijun and tenkan_sen[i-1] >= kijun_sen[i-1]
        
        # Price relative to cloud
        price_above_cloud = close_val > upper_cloud
        price_below_cloud = close_val < lower_cloud
        price_in_cloud = lower_cloud <= close_val <= upper_cloud
        
        if position == 0:
            # Flat - look for entry
            # Long: TK cross up, price above cloud, and above weekly EMA (uptrend)
            long_entry = tk_cross_up and price_above_cloud and (close_val > ema_21_w)
            # Short: TK cross down, price below cloud, and below weekly EMA (downtrend)
            short_entry = tk_cross_down and price_below_cloud and (close_val < ema_21_w)
            
            if long_entry:
                signals[i] = fixed_size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -fixed_size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on TK cross down or price drops below cloud
            if tk_cross_down or close_val < lower_cloud:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = fixed_size
        elif position == -1:
            # Short - exit on TK cross up or price rises above cloud
            if tk_cross_up or close_val > upper_cloud:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -fixed_size
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1wTrend"
timeframe = "6h"
leverage = 1.0