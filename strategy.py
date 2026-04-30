#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Ichimoku cloud filter with TK cross timing and volume confirmation
# Ichimoku cloud provides dynamic support/resistance and trend direction from higher timeframe
# TK cross (Tenkan-Kijun) gives precise entry timing within the 6h chart
# Volume confirmation ensures institutional participation
# Designed for low trade frequency (<30/year) to minimize fee drag in both bull and bear markets

name = "6h_Ichimoku_TK_Cross_12hCloudFilter_VolumeSpike_v1"
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
    
    # Load 12h data ONCE before loop for Ichimoku calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 52:  # Need enough for Senkou Span B (52 periods)
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Ichimoku components on 12h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    tenkan_sen = (pd.Series(high_12h).rolling(window=period_tenkan, min_periods=period_tenkan).max() + 
                  pd.Series(low_12h).rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    kijun_sen = (pd.Series(high_12h).rolling(window=period_kijun, min_periods=period_kijun).max() + 
                 pd.Series(low_12h).rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(period_kijun)  # shifted 26 periods ahead
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    senkou_span_b = ((pd.Series(high_12h).rolling(window=period_senkou_b, min_periods=period_senkou_b).max() + 
                      pd.Series(low_12h).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2).shift(period_kijun)
    
    # Align Ichimoku components to 6h timeframe (wait for completed 12h bar)
    tenkan_aligned = align_htf_to_ltf(prices, df_12h, tenkan_sen.values)
    kijun_aligned = align_htf_to_ltf(prices, df_12h, kijun_sen.values)
    senkou_a_aligned = align_htf_to_ltf(prices, df_12h, senkou_span_a.values)
    senkou_b_aligned = align_htf_to_ltf(prices, df_12h, senkou_span_b.values)
    
    # Calculate ATR(14) for dynamic stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 52  # warmup for Ichimoku calculation
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 2.0x 30-period average
        vol_ma_30 = np.mean(volume[max(0, i-30):i])
        volume_spike = volume[i] > (2.0 * vol_ma_30)
        
        curr_close = close[i]
        curr_tenkan = tenkan_aligned[i]
        curr_kijun = kijun_aligned[i]
        curr_senkou_a = senkou_a_aligned[i]
        curr_senkou_b = senkou_b_aligned[i]
        curr_atr = atr[i]
        
        # Determine cloud top and bottom
        cloud_top = max(curr_senkou_a, curr_senkou_b)
        cloud_bottom = min(curr_senkou_a, curr_senkou_b)
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if volume_spike:
                # Bullish entry: TK cross bullish + price above cloud
                if curr_tenkan > curr_kijun and curr_close > cloud_top:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: TK cross bearish + price below cloud
                elif curr_tenkan < curr_kijun and curr_close < cloud_bottom:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.5 * ATR below entry price OR price breaks below cloud bottom
            if curr_close < entry_price - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < cloud_bottom:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches opposite side of cloud
            elif curr_close >= cloud_top:
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.5 * ATR above entry price OR price breaks above cloud top
            if curr_close > entry_price + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > cloud_top:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches opposite side of cloud
            elif curr_close <= cloud_bottom:
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.25
    
    return signals