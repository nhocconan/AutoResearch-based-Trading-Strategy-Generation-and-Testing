#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation
# Uses Tenkan/Kijun cross + price relative to Senkou Span cloud for trend entries
# 1d ADX > 25 ensures trades align with strong daily trend to avoid chop
# Volume spike (1.8x 20-period average) confirms institutional participation
# Discrete sizing 0.28 targets 50-120 trades over 4 years (12-30/year)
# Works in bull/bear by only taking breakouts in direction of 1d trend

name = "6h_Ichimoku_1dADX25_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min()
    tenkan_sen = ((period9_high + period9_low) / 2).values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min()
    kijun_sen = ((period26_high + period26_low) / 2).values
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min()
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Calculate ADX(14) on 1d for trend strength
    plus_dm = pd.Series(df_1d['high']).diff()
    minus_dm = pd.Series(df_1d['low']).diff().copy()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = abs(minus_dm)
    
    tr1 = pd.Series(df_1d['high']) - pd.Series(df_1d['low'])
    tr2 = abs(pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift(1))
    tr3 = abs(pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(window=14, min_periods=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14, min_periods=14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14, min_periods=14).mean() / atr)
    dx = (abs(plus_di - minus_di) / (abs(plus_di + minus_di))) * 100
    adx = dx.rolling(window=14, min_periods=14).mean().values
    
    # Align Ichimoku components and ADX to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a, additional_delay_bars=26)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b, additional_delay_bars=26)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di.values)
    minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di.values)
    
    # Volume confirmation (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Ichimoku calculations)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud color and position
        # When Senkou A > Senkou B: bullish cloud (green)
        # When Senkou A < Senkou B: bearish cloud (red)
        bullish_cloud = senkou_a_aligned[i] > senkou_b_aligned[i]
        bearish_cloud = senkou_a_aligned[i] < senkou_b_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Ichimoku bullish signal: Tenkan crosses above Kijun AND price above cloud
            tenkan_cross_up = tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]
            price_above_close = close[i] > senkou_a_aligned[i] and close[i] > senkou_b_aligned[i]
            
            # Ichimoku bearish signal: Tenkan crosses below Kijun AND price below cloud
            tenkan_cross_down = tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1]
            price_below_close = close[i] < senkou_a_aligned[i] and close[i] < senkou_b_aligned[i]
            
            # 1d ADX trend filter: ADX > 25 indicates strong trend
            adx_strong = adx_aligned[i] > 25
            # Get 1d DI values for trend direction
            adx_long = adx_strong and (plus_di_aligned[i] > minus_di_aligned[i])
            adx_short = adx_strong and (minus_di_aligned[i] > plus_di_aligned[i])
            
            # Long entry: bullish TK cross + price above bullish cloud + 1d uptrend + volume
            if tenkan_cross_up and price_above_close and bullish_cloud and adx_long and volume_spike[i]:
                signals[i] = 0.28
                position = 1
            # Short entry: bearish TK cross + price below bearish cloud + 1d downtrend + volume
            elif tenkan_cross_down and price_below_close and bearish_cloud and adx_short and volume_spike[i]:
                signals[i] = -0.28
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Tenkan crosses below Kijun OR price falls below cloud OR ADX weakens
            tenkan_cross_down = tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1]
            price_below_cloud = close[i] < senkou_a_aligned[i] or close[i] < senkou_b_aligned[i]
            adx_weak = adx_aligned[i] < 20
            
            if tenkan_cross_down or price_below_cloud or adx_weak:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        
        elif position == -1:  # Short position
            # Exit: Tenkan crosses above Kijun OR price rises above cloud OR ADX weakens
            tenkan_cross_up = tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]
            price_above_cloud = close[i] > senkou_a_aligned[i] and close[i] > senkou_b_aligned[i]
            adx_weak = adx_aligned[i] < 20
            
            if tenkan_cross_up or price_above_cloud or adx_weak:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals