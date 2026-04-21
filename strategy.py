# 12h_IchimokuKumo_TenkanKijun_Cross_1dTrend_Volume
# Hypothesis: 12h strategy using Ichimoku cloud (Kumo) breakout with 1d EMA trend filter and volume confirmation.
# In uptrend (price > 1d EMA50), buy when Tenkan-sen crosses above Kijun-sen AND price is above Kumo.
# In downtrend (price < 1d EMA50), sell when Tenkan-sen crosses below Kijun-sen AND price is below Kumo.
# Volume must exceed 1.8x 20-period average to confirm signal strength.
# Exit on Tenkan/Kijun cross reversal or when price re-enters the Kumo.
# Designed for 12-37 trades/year (50-150 total over 4 years) to minimize fee drag while capturing major trend moves.
# Ichimoku provides multi-component trend/momentum signal proven effective in crypto trends.
# Works in bull markets via bullish crosses above cloud and in bear markets via bearish crosses below cloud.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for Ichimoku components
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 52:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_12h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_12h).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_12h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_12h).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_12h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_12h).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Kumo (Cloud) boundaries: Senkou Span A and B shifted forward 26 periods
    # For look-ahead avoidance, we use the cloud values from 26 periods ago (already formed)
    senkou_span_a_lagged = np.roll(senkou_span_a, 26)
    senkou_span_b_lagged = np.roll(senkou_span_b, 26)
    # Set first 26 values to NaN since they depend on future data
    senkou_span_a_lagged[:26] = np.nan
    senkou_span_b_lagged[:26] = np.nan
    
    # Kumo top and bottom (for price vs cloud checks)
    kumo_top = np.maximum(senkou_span_a_lagged, senkou_span_b_lagged)
    kumo_bottom = np.minimum(senkou_span_a_lagged, senkou_span_b_lagged)
    
    # Align 12h Ichimoku components to 12h timeframe (no further alignment needed as we're using 12h data)
    # But we need to align to the close of each 12h bar
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_12h, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_12h, kijun_sen)
    kumo_top_aligned = align_htf_to_ltf(prices, df_12h, kumo_top)
    kumo_bottom_aligned = align_htf_to_ltf(prices, df_12h, kumo_bottom)
    
    # Load 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation (volume spike > 1.8x 20-period average)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(kumo_top_aligned[i]) or np.isnan(kumo_bottom_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        tenkan = tenkan_sen_aligned[i]
        kijun = kijun_sen_aligned[i]
        kumo_top_val = kumo_top_aligned[i]
        kumo_bottom_val = kumo_bottom_aligned[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Kumo twist detection (Senkou A vs B)
        kumo_twist = tenkan > kijun  # Simplified: bullish when Tenkan > Kijun
        
        if position == 0:
            # Enter long: bullish TK cross AND price above Kumo AND uptrend (price > 1d EMA50) AND volume spike
            if (tenkan > kijun and  # Bullish TK cross
                price_close > kumo_top_val and  # Price above Kumo
                price_close > ema_trend and     # Uptrend filter
                vol_ratio_val > 1.8):           # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Enter short: bearish TK cross AND price below Kumo AND downtrend (price < 1d EMA50) AND volume spike
            elif (tenkan < kijun and    # Bearish TK cross
                  price_close < kumo_bottom_val and  # Price below Kumo
                  price_close < ema_trend and        # Downtrend filter
                  vol_ratio_val > 1.8):              # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: TK cross reversal OR price re-enters Kumo
            exit_signal = False
            
            # TK cross reversal exit
            if position == 1 and tenkan < kijun:  # Bullish to bearish cross
                exit_signal = True
            elif position == -1 and tenkan > kijun:  # Bearish to bullish cross
                exit_signal = True
            
            # Price re-enters Kumo exit
            if position == 1 and price_close < kumo_top_val:  # Long exits if price drops below cloud top
                exit_signal = True
            elif position == -1 and price_close > kumo_bottom_val:  # Short exits if price rises above cloud bottom
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_IchimokuKumo_TenkanKijun_Cross_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0