#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation
# Uses Kumo (cloud) from 1d as primary trend filter, TK cross on 6h for entry timing
# Volume spike confirms breakout strength. Works in bull/bear via cloud direction.
# Target: 50-150 total trades over 4 years (12-37/year). Discrete sizing 0.25.

name = "6h_Ichimoku_Kumo_TK_Cross_1dTrend_Volume_v1"
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
    
    # Calculate 1d Ichimoku components (TK cross + Kumo)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 52 periods for Senkou Span B
        return np.zeros(n)
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    tenkan_sen = (pd.Series(df_1d['high']).rolling(window=period_tenkan, min_periods=period_tenkan).max() +
                  pd.Series(df_1d['low']).rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    kijun_sen = (pd.Series(df_1d['high']).rolling(window=period_kijun, min_periods=period_kijun).max() +
                 pd.Series(df_1d['low']).rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    senkou_span_b = ((pd.Series(df_1d['high']).rolling(window=period_senkou_b, min_periods=period_senkou_b).max() +
                      pd.Series(df_1d['low']).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2).shift(26)
    
    # Align Ichimoku components to 6h timeframe (wait for completed 1d bar)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    # Calculate 6h TK cross (Tenkan/Kijun cross)
    # Tenkan-sen on 6h for entry timing
    tenkan_sen_6h = (pd.Series(high).rolling(window=9, min_periods=9).max() +
                     pd.Series(low).rolling(window=9, min_periods=9).min()) / 2
    kijun_sen_6h = (pd.Series(high).rolling(window=26, min_periods=26).max() +
                    pd.Series(low).rolling(window=26, min_periods=26).min()) / 2
    tk_cross = tenkan_sen_6h - kijun_sen_6h  # >0 = bullish cross
    
    # Volume confirmation: volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 52, 26, 20, 14)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(tk_cross[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_tenkan_1d = tenkan_sen_aligned[i]
        curr_kijun_1d = kijun_sen_aligned[i]
        curr_senkou_a = senkou_span_a_aligned[i]
        curr_senkou_b = senkou_span_b_aligned[i]
        curr_tk_cross = tk_cross[i]
        curr_volume_spike = volume_spike[i]
        curr_atr = atr_14[i]
        
        # Kumo (cloud) boundaries and color
        upper_kumo = max(curr_senkou_a, curr_senkou_b)
        lower_kumo = min(curr_senkou_a, curr_senkou_b)
        kumo_bullish = curr_senkou_a > curr_senkou_b  # Senkou A above B = bullish cloud
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with TK cross aligned with Kumo trend
            if curr_volume_spike:
                # Bullish: TK cross bullish + price above Kumo + Kumo bullish
                if curr_tk_cross > 0 and curr_close > upper_kumo and kumo_bullish:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish: TK cross bearish + price below Kumo + Kumo bearish
                elif curr_tk_cross < 0 and curr_close < lower_kumo and not kumo_bullish:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2 * ATR below entry OR price drops below Kumo
            stop_loss = entry_price - 2.0 * curr_atr
            # Exit: Stoploss hit OR price closes below Kumo OR Kumo turns bearish
            if curr_low <= stop_loss or curr_close < lower_kumo or not kumo_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry OR price rises above Kumo
            stop_loss = entry_price + 2.0 * curr_atr
            # Exit: Stoploss hit OR price closes above Kumo OR Kumo turns bullish
            if curr_high >= stop_loss or curr_close > upper_kumo or kumo_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals