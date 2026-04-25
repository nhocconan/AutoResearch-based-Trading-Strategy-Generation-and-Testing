#!/usr/bin/env python3
"""
4h Camarilla H3L3 Breakout + Volume Spike + 1d EMA34 Trend Filter + Chop Filter
Hypothesis: Camarilla pivot levels act as intraday support/resistance. Breakouts above H3 or below L3 with volume confirmation indicate institutional participation. 1d EMA34 filter ensures trades align with daily trend, reducing false breakouts in choppy markets. Chop filter avoids ranging markets. Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 20-40 trades/year on 4h.
Works in bull markets via breakouts with trend and in bear markets via trend filter (avoids counter-trend entries) and chop filter (avoids whipsaws).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot calculation and EMA trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivots from previous 1d OHLC
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    rang = prev_high - prev_low
    H3 = prev_close + 1.0 * rang
    L3 = prev_close - 1.0 * rang
    
    # Align Camarilla levels to 4h (use previous day's levels for current day's trading)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Calculate Choppiness Index (CHOP) on 1d to filter ranging markets
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (max(high,n) - min(low,n))))
    # Simplified: use 14-period high-low range vs ATR
    high_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    low_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    tr1 = pd.Series(df_1d['high']) - pd.Series(df_1d['low'])
    tr2 = abs(pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift(1))
    tr3 = abs(pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14, min_periods=14).mean().values
    chop = 100 * np.log10(np.sum(tr[-14:], axis=0) / (np.log10(14) * (high_14 - low_14))) if len(df_1d) >= 14 else np.full_like(close, 50.0)
    # For simplicity, use a proxy: chop > 61.8 = ranging, chop < 38.2 = trending
    # We'll calculate a rolling chop value for each 1d bar
    chop_values = []
    for j in range(len(df_1d)):
        if j < 13:
            chop_values.append(50.0)  # neutral
        else:
            high_14 = np.max(df_1d['high'].values[j-13:j+1])
            low_14 = np.min(df_1d['low'].values[j-13:j+1])
            atr_14 = np.mean(np.abs(df_1d['high'].values[j-13:j+1] - df_1d['low'].values[j-13:j+1]))
            if atr_14 == 0 or (high_14 - low_14) == 0:
                chop_values.append(50.0)
            else:
                chop_val = 100 * np.log10(atr_14 * 14 / (np.log10(14) * (high_14 - low_14)))
                chop_values.append(chop_val)
    chop_values = np.array(chop_values)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    # Chop filter: only trade when market is trending (CHOP < 38.2)
    chop_filter = chop_aligned < 38.2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for volume MA and EMA
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        H3_level = H3_aligned[i]
        L3_level = L3_aligned[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        is_trending = chop_filter[i]
        
        if position == 0:
            # Look for entry signals (only in trending markets)
            # Long: price breaks above H3 AND volume spike AND price > 1d EMA34 (uptrend) AND trending market
            long_entry = (curr_close > H3_level) and vol_spike and (curr_close > ema_trend) and is_trending
            # Short: price breaks below L3 AND volume spike AND price < 1d EMA34 (downtrend) AND trending market
            short_entry = (curr_close < L3_level) and vol_spike and (curr_close < ema_trend) and is_trending
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below L3 (reversal) OR price < 1d EMA34 (trend change) OR chop becomes too high (ranging)
            if (curr_close < L3_level) or (curr_close < ema_trend) or (chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above H3 (reversal) OR price > 1d EMA34 (trend change) OR chop becomes too high (ranging)
            if (curr_close > H3_level) or (curr_close > ema_trend) or (chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_VolumeSpike_1dEMA34_Trend_ChopFilter"
timeframe = "4h"
leverage = 1.0