#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h ATR-based volume spike and 1d choppiness regime filter
# - Primary: 4h price breaks above Donchian(20) high (long) or below Donchian(20) low (short)
# - Volume filter: 12h ATR(10) > 1.5x 20-period ATR MA to confirm institutional participation (more reliable than raw volume)
# - Regime filter: 1d Choppiness Index(14) > 61.8 to avoid choppy/ranging markets (focus on trending conditions)
# - Entry: Long when breakout above upper band + volume spike + CHOP > 61.8
#          Short when breakout below lower band + volume spike + CHOP > 61.8
# - Exit: Close crosses back inside Donchian channel (mean reversion exit)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# - Works in bull/bear: Donchian breakouts capture trends, ATR volume spike avoids fakeouts, CHOP filter avoids false signals in ranging markets

name = "4h_12h_1d_donchian_atr_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period) on 4h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h ATR(10) for volume spike filter
    high_low_12h = high_12h - low_12h
    high_close_12h = np.abs(high_12h - np.roll(close_12h, 1))
    low_close_12h = np.abs(low_12h - np.roll(close_12h, 1))
    high_close_12h[0] = high_low_12h[0]
    low_close_12h[0] = high_low_12h[0]
    tr_12h = np.maximum(high_low_12h, np.maximum(high_close_12h, low_close_12h))
    atr_10_12h = pd.Series(tr_12h).rolling(window=10, min_periods=10).mean().values
    atr_ma_20_12h = pd.Series(atr_10_12h).rolling(window=20, min_periods=20).mean().values
    atr_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_ma_20_12h)
    
    # Calculate 1d Choppiness Index(14)
    # CHOP = 100 * log10(sum(TR(14)) / (log10(14) * (max(HH(14)) - min(LL(14)))))
    high_low_1d = high_1d - low_1d
    high_close_1d = np.abs(high_1d - np.roll(close_1d, 1))
    low_close_1d = np.abs(low_1d - np.roll(close_1d, 1))
    high_close_1d[0] = high_low_1d[0]
    low_close_1d[0] = high_low_1d[0]
    tr_1d = np.maximum(high_low_1d, np.maximum(high_close_1d, low_close_1d))
    tr_sum_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = np.log10(14) * (hh_14 - ll_14)
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # avoid division by zero
    chop_14 = 100 * np.log10(tr_sum_14 / chop_denom)
    chop_14_aligned = align_htf_to_ltf(prices, df_1d, chop_14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(30, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(atr_ma_20_12h_aligned[i]) or np.isnan(chop_14_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume spike filter: current 12h ATR(10) > 1.5x 20-period ATR MA
        atr_10_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_10_12h)
        volume_spike = atr_10_12h_aligned[i] > 1.5 * atr_ma_20_12h_aligned[i]
        
        # Regime filter: CHOP > 61.8 indicates trending market (avoid chop)
        regime_filter = chop_14_aligned[i] > 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long entry: breakout above upper Donchian band + volume spike + CHOP > 61.8
            if (close[i] > highest_20[i] and volume_spike and regime_filter):
                position = 1
                signals[i] = 0.25
            # Short entry: breakout below lower Donchian band + volume spike + CHOP > 61.8
            elif (close[i] < lowest_20[i] and volume_spike and regime_filter):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Close crosses back inside Donchian channel (mean reversion exit)
            if position == 1:  # Long position
                if close[i] < lowest_20[i]:  # Exit when price breaks below lower band
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] > highest_20[i]:  # Exit when price breaks above upper band
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals