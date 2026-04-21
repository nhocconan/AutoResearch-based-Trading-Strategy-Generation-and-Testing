#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1dCloudFilter_VolumeSpike_v1
Hypothesis: 6h Ichimoku Tenkan-Kijun (TK) cross with 1d cloud filter (price above/below cloud) and volume confirmation (>2.0x 20-bar MA). 
Long when TK cross bullish + price above 1d cloud + volume spike. Short when TK cross bearish + price below 1d cloud + volume spike. 
ATR-based stoploss (2.5x) and discrete sizing (0.25) to control risk and churn. Target: 50-150 total trades over 4 years by requiring confluence of TK cross, 1d cloud alignment, and strong volume. 
Designed to work in bull (TK bullish + above cloud) and bear (TK bearish + below cloud) regimes with strict volume filter to avoid overtrading.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for cloud, 1w for EMA200 trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Ichimoku cloud (Senkou Span A/B) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2.0)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2.0)
    
    # Align Ichimoku components to 6h timeframe (no extra delay needed as they are based on completed 1d candles)
    tenkan_sen_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_6h = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_6h = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_6h = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # === 1d EMA200 for higher timeframe trend filter (needs extra delay as it's a trend indicator) ===
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_6h = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === 6h ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === 6h volume confirmation (volume > 2.0x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i]) or 
            np.isnan(senkou_span_a_6h[i]) or np.isnan(senkou_span_b_6h[i]) or
            np.isnan(ema_200_6h[i]) or np.isnan(atr[i]) or 
            np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        tenkan = tenkan_sen_6h[i]
        kijun = kijun_sen_6h[i]
        span_a = senkou_span_a_6h[i]
        span_b = senkou_span_b_6h[i]
        ema_200 = ema_200_6h[i]
        vol_conf = volume_confirmed[i]
        
        # Cloud boundaries (top and bottom of cloud)
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        # TK cross signals (bullish when Tenkan > Kijun, bearish when Tenkan < Kijun)
        tk_bullish = tenkan > kijun
        tk_bearish = tenkan < kijun
        
        # Price relative to cloud
        price_above_cloud = price > cloud_top
        price_below_cloud = price < cloud_bottom
        
        # Higher timeframe trend filter (1d EMA200)
        price_above_200ema = price > ema_200
        price_below_200ema = price < ema_200
        
        if position == 0:
            # Long conditions: TK bullish + price above cloud + above 200 EMA + volume spike
            long_condition = tk_bullish and price_above_cloud and price_above_200ema and vol_conf
            # Short conditions: TK bearish + price below cloud + below 200 EMA + volume spike
            short_condition = tk_bearish and price_below_cloud and price_below_200ema and vol_conf
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Check stoploss (2.5x ATR)
            if position == 1:
                if price < entry_price - 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if TK cross turns bearish or price falls below cloud
                elif not tk_bullish or price < cloud_bottom:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if TK cross turns bullish or price rises above cloud
                elif not tk_bearish or price > cloud_top:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_1dCloudFilter_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0