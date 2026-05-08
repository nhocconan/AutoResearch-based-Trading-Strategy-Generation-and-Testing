#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with daily filter and volume confirmation
# Uses Ichimoku system: Tenkan-sen (9), Kijun-sen (26), Senkou Span A/B (26, 52)
# Long when: price above cloud + TK cross bullish + daily trend up + volume spike
# Short when: price below cloud + TK cross bearish + daily trend down + volume spike
# Ichimoku provides dynamic support/resistance and trend direction, effective in both bull/bear markets
# Target: 15-35 trades/year to minimize fee drag while capturing major trends

name = "6h_Ichimoku_Cloud_DailyTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Ichimoku components (6h)
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    max_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (max_high_senkou_b + min_low_senkou_b) / 2
    
    # Daily trend filter (1d EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for Ichimoku (52 + 26 shift)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud boundaries (shifted values for current price)
        senkou_a_shifted = senkou_a[i - 26] if i >= 26 else senkou_a[i]
        senkou_b_shifted = senkou_b[i - 26] if i >= 26 else senkou_b[i]
        
        # Cloud top and bottom
        cloud_top = max(senkou_a_shifted, senkou_b_shifted)
        cloud_bottom = min(senkou_a_shifted, senkou_b_shifted)
        
        # TK cross
        tk_cross_bullish = tenkan[i] > kijun[i]
        tk_cross_bearish = tenkan[i] < kijun[i]
        
        if position == 0:
            # Long: price above cloud + TK bullish + daily uptrend + volume spike
            price_above_cloud = close[i] > cloud_top
            daily_uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            long_cond = price_above_cloud and tk_cross_bullish and daily_uptrend and volume_spike[i]
            
            # Short: price below cloud + TK bearish + daily downtrend + volume spike
            price_below_cloud = close[i] < cloud_bottom
            daily_downtrend = ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]
            
            short_cond = price_below_cloud and tk_cross_bearish and daily_downtrend and volume_spike[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below cloud or TK cross bearish
            price_below_cloud = close[i] < cloud_bottom
            tk_cross_bearish = tenkan[i] < kijun[i]
            
            if price_below_cloud or tk_cross_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above cloud or TK cross bullish
            price_above_cloud = close[i] > cloud_top
            tk_cross_bullish = tenkan[i] > kijun[i]
            
            if price_above_cloud or tk_cross_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals