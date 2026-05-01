#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud + TK Cross with 1d ADX trend filter and volume confirmation.
# Uses Ichimoku from 1d timeframe for major trend structure, TK cross on 6h for entry timing.
# Long when: price > 1d Ichimoku cloud, TK cross bullish, ADX > 25, volume > 1.5x 20-period median.
# Short when: price < 1d Ichimoku cloud, TK cross bearish, ADX > 25, volume > 1.5x 20-period median.
# Ichimoku cloud acts as dynamic support/resistance, reducing false signals in ranging markets.
# ADX filter ensures we only trade in trending conditions, avoiding whipsaws in chop.
# Volume confirmation adds conviction to breakouts.
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years).
# Discrete sizing: 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.5*ATR.

name = "6h_Ichimoku_TK_Cross_1dADX_Volume_v1"
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
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate 1d Ichimoku components (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # need 26*2 for Ichimoku
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_senkou + min_low_senkou) / 2)
    
    # Align Ichimoku components to 6h timeframe (with proper delay for forward plots)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a, additional_delay_bars=26)  # plotted 26 periods ahead
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b, additional_delay_bars=26)  # plotted 26 periods ahead
    
    # Calculate 1d ADX trend filter (HTF)
    # ADX calculation: +DI, -DI, DX
    period_adx = 14
    # True Range
    tr_1d = np.maximum(
        np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1])),
        np.abs(low_1d[1:] - close_1d[:-1])
    )
    tr_1d = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], tr_1d])
    
    # +DM and -DM
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_period_sum = pd.Series(tr_1d).rolling(window=period_adx, min_periods=period_adx).sum().values
    dm_plus_sum = pd.Series(dm_plus).rolling(window=period_adx, min_periods=period_adx).sum().values
    dm_minus_sum = pd.Series(dm_minus).rolling(window=period_adx, min_periods=period_adx).sum().values
    
    # +DI and -DI
    di_plus = 100 * dm_plus_sum / tr_period_sum
    di_minus = 100 * dm_minus_sum / tr_period_sum
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=period_adx, min_periods=period_adx).mean().values
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, volume, Ichimoku, and ADX
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(vol_median_20[i]) or 
            np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or 
            np.isnan(senkou_b_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Ichimoku cloud: price above/below cloud
        cloud_top = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        price_above_cloud = curr_close > cloud_top
        price_below_cloud = curr_close < cloud_bottom
        
        # TK Cross: Tenkan-sen crossing Kijun-sen
        # Use previous bar to avoid look-ahead in cross detection
        if i > 0:
            tenkan_prev = tenkan_aligned[i-1]
            kijun_prev = kijun_aligned[i-1]
            tk_bullish = tenkan_aligned[i] > kijun_aligned[i] and tenkan_prev <= kijun_prev
            tk_bearish = tenkan_aligned[i] < kijun_aligned[i] and tenkan_prev >= kijun_prev
        else:
            tk_bullish = False
            tk_bearish = False
        
        # Trend filter: ADX > 25 indicates trending market
        strong_trend = adx_aligned[i] > 25
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: price above cloud, bullish TK cross, strong trend, volume spike
            if price_above_cloud and tk_bullish and strong_trend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price below cloud, bearish TK cross, strong trend, volume spike
            elif price_below_cloud and tk_bearish and strong_trend and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below cloud OR TK cross turns bearish OR trend weakens
            elif curr_close < cloud_bottom or not tk_bullish or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above cloud OR TK cross turns bullish OR trend weakens
            elif curr_close > cloud_top or not tk_bearish or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals