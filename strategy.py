#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + Donchian(20) breakout with volume confirmation.
# In choppy markets (CHOP > 61.8): mean revert at Bollinger Bands (20,2).
# In trending markets (CHOP < 38.2): breakout Donchian(20) with volume > 1.5x average.
# Uses 12h trend filter (EMA50) to avoid counter-trend trades.
# Designed for ~20-30 trades/year with strict regime-dependent entries.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR(1)) / (n * ATR(14))) / log10(n)
    # Simplified: CHOP = 100 * log10( sum(tr) over 14 / (ATR(14) * 14) ) / log10(14)
    tr = np.maximum(high[1:] - low[1:], np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])  # align with high/low indices
    
    atr_1 = tr
    sum_tr_14 = np.full(n, np.nan)
    atr_14 = np.full(n, np.nan)
    
    for i in range(13, n):
        sum_tr_14[i] = np.nansum(tr[i-13:i+1])
        atr_14[i] = np.nanmean(tr[i-13:i+1])
    
    chop = np.full(n, np.nan)
    for i in range(13, n):
        if not np.isnan(sum_tr_14[i]) and not np.isnan(atr_14[i]) and atr_14[i] > 0:
            chop[i] = 100 * np.log10(sum_tr_14[i] / (atr_14[i] * 14)) / np.log10(14)
    
    # Calculate Bollinger Bands (20,2) for mean reversion in chop
    close_s = pd.Series(close)
    bb_mid = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    
    # Calculate Donchian channels (20) for breakout in trend
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 20-period indicators
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(chop[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend filter from 12h EMA50
        bullish_trend = price > ema50_12h_aligned[i]
        bearish_trend = price < ema50_12h_aligned[i]
        
        # Regime filters
        is_choppy = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        
        if position == 0:
            if is_choppy:
                # Mean reversion in chop: fade at Bollinger Bands
                if price <= bb_lower[i] and vol_filter:
                    signals[i] = size
                    position = 1
                elif price >= bb_upper[i] and vol_filter:
                    signals[i] = -size
                    position = -1
                else:
                    signals[i] = 0.0
            elif is_trending:
                # Breakout in trend: Donchian breakout with volume
                if price > donchian_high[i] and vol_filter and bullish_trend:
                    signals[i] = size
                    position = 1
                elif price < donchian_low[i] and vol_filter and bearish_trend:
                    signals[i] = -size
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Neutral chop: no trade
                signals[i] = 0.0
        elif position == 1:
            # Exit long: 
            # In chop: return to BB middle
            # In trend: Donchian low break or trend reversal
            if is_choppy:
                if price >= bb_mid[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = size
            else:  # trending
                if price < donchian_low[i] or not bullish_trend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = size
        elif position == -1:
            # Exit short:
            # In chop: return to BB middle
            # In trend: Donchian high break or trend reversal
            if is_choppy:
                if price <= bb_mid[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -size
            else:  # trending
                if price > donchian_high[i] or not bearish_trend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -size
    
    return signals

name = "4h_Chop_DonchianBB_Volume_12hTrend"
timeframe = "4h"
leverage = 1.0