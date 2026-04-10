#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation
# - Uses Ichimoku components: Tenkan-sen (9), Kijun-sen (26), Senkou Span A/B (52 displacement)
# - Long when price > Kumo (cloud) AND Tenkan > Kijun (bullish TK cross) in 1d uptrend with volume spike
# - Short when price < Kumo AND Tenkan < Kijun (bearish TK cross) in 1d downtrend with volume spike
# - Uses discrete position sizing (0.25) to minimize fee churn
# - ATR-based stoploss: exit when price moves against position by 2.5x ATR(14) or TK cross reverses
# - Targets 12-30 trades/year (50-120 total over 4 years) to avoid fee drag
# - Ichimoku works in both bull/bear markets: cloud acts as dynamic S/R, TK cross captures momentum

name = "6h_1d_ichimoku_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d Ichimoku Cloud calculation
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Current Kumo (cloud) boundaries: Senkou Span A/B shifted back 26 periods
    # For point i, cloud is based on data from i-26
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Cloud top/bottom
    cloud_top = np.maximum(senkou_a_shifted, senkou_b_shifted)
    cloud_bottom = np.minimum(senkou_a_shifted, senkou_b_shifted)
    
    # TK Cross signals
    tk_bullish = tenkan > kijun  # Bullish when Tenkan > Kijun
    tk_bearish = tenkan < kijun  # Bearish when Tenkan < Kijun
    
    # 1d EMA(200) for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # 1d ATR(14) for stoploss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_1d = np.zeros_like(tr)
    atr_14_1d[14-1] = np.mean(tr[:14])
    for i in range(14, len(tr)):
        atr_14_1d[i] = (atr_14_1d[i-1] * (14-1) + tr[i]) / 14
    
    # 1d volume confirmation: > 1.5x 20-period average
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * avg_volume_20_1d)
    
    # Align all HTF indicators to LTF (6h)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    cloud_top_aligned = align_htf_to_ltf(prices, df_1d, cloud_top)
    cloud_bottom_aligned = align_htf_to_ltf(prices, df_1d, cloud_bottom)
    tk_bullish_aligned = align_htf_to_ltf(prices, df_1d, tk_bullish)
    tk_bearish_aligned = align_htf_to_ltf(prices, df_1d, tk_bearish)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(cloud_top_aligned[i]) or np.isnan(cloud_bottom_aligned[i]) or
            np.isnan(tk_bullish_aligned[i]) or np.isnan(tk_bearish_aligned[i]) or
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(vol_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss or TK cross turns bearish
            if (prices['close'].iloc[i] < entry_price - 2.5 * entry_atr or 
                not tk_bullish_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss or TK cross turns bullish
            if (prices['close'].iloc[i] > entry_price + 2.5 * entry_atr or 
                not tk_bearish_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Ichimoku signals with trend and volume filters
            if vol_spike_1d_aligned[i]:
                price = prices['close'].iloc[i]
                # Long signal: price above cloud AND bullish TK cross in 1d uptrend
                if (price > cloud_top_aligned[i] and 
                    tk_bullish_aligned[i] and 
                    price > ema_200_1d_aligned[i]):
                    position = 1
                    entry_price = price
                    entry_atr = atr_14_1d_aligned[i]
                    signals[i] = 0.25
                # Short signal: price below cloud AND bearish TK cross in 1d downtrend
                elif (price < cloud_bottom_aligned[i] and 
                      tk_bearish_aligned[i] and 
                      price < ema_200_1d_aligned[i]):
                    position = -1
                    entry_price = price
                    entry_atr = atr_14_1d_aligned[i]
                    signals[i] = -0.25
    
    return signals