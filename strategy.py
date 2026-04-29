#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation
# Uses Ichimoku components (Tenkan-sen, Kijun-sen, Senkou Span A/B) from 6h data
# Long when price > cloud AND Tenkan > Kijun AND 1d EMA50 uptrend
# Short when price < cloud AND Tenkan < Kijun AND 1d EMA50 downtrend
# Volume confirmation reduces false breaks. Works in both bull/bear by following 1d trend.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "6h_Ichimoku_1dEMA50_Trend_VolumeSpike_v1"
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
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku Cloud (9, 26, 52) on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    highest_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    lowest_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (highest_high_tenkan + lowest_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    highest_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    lowest_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (highest_high_kijun + lowest_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    highest_high_senkou = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    lowest_low_senkou = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (highest_high_senkou + lowest_low_senkou) / 2
    
    # The cloud is between Senkou Span A and Senkou Span B
    # For plotting, Senkou spans are shifted 26 periods ahead
    # For trading, we compare current price to current cloud (unshifted)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 50, 20, 26)  # warmup for Senkou B, EMA50, volume MA, Kijun
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema50_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_tenkan = tenkan[i]
        curr_kijun = kijun[i]
        curr_senkou_a = senkou_a[i]
        curr_senkou_b = senkou_b[i]
        curr_ema50 = ema50_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Cloud boundaries (Senkou Span A and B form the cloud)
        cloud_top = max(curr_senkou_a, curr_senkou_b)
        cloud_bottom = min(curr_senkou_a, curr_senkou_b)
        
        # Trend regime: bullish if price > 1d EMA50, bearish if price < 1d EMA50
        is_bullish_regime = curr_close > curr_ema50
        is_bearish_regime = curr_close < curr_ema50
        
        # Ichimoku signals
        price_above_cloud = curr_close > cloud_top
        price_below_cloud = curr_close < cloud_bottom
        tenkan_above_kijun = curr_tenkan > curr_kijun
        tenkan_below_kijun = curr_tenkan < curr_kijun
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Bullish entry: price > cloud AND Tenkan > Kijun AND bullish regime
                if price_above_cloud and tenkan_above_kijun and is_bullish_regime:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price < cloud AND Tenkan < Kijun AND bearish regime
                elif price_below_cloud and tenkan_below_kijun and is_bearish_regime:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price < cloud OR Tenkan < Kijun OR regime changes to bearish
            if (not price_above_cloud) or (not tenkan_above_kijun) or (not is_bullish_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price > cloud OR Tenkan > Kijun OR regime changes to bullish
            if (not price_below_cloud) or (not tenkan_below_kijun) or (not is_bearish_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals