#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume spike + 1w trend filter (EMA50 > EMA200)
# - Long: price > Donchian high(20) AND 1d volume > 2.0x 20-period average AND 1w EMA50 > 1w EMA200
# - Short: price < Donchian low(20) AND 1d volume > 2.0x 20-period average AND 1w EMA50 < 1w EMA200
# - Uses discrete position sizing (0.25) to minimize fee churn
# - ATR-based stoploss (2.0x ATR(14)) and Donchian opposite exit for risk management
# - Designed for 4h timeframe: targets 20-40 trades/year to avoid fee drag
# - Volume spike ensures momentum confirmation, weekly EMA filter ensures we trade with the higher timeframe trend
# - Works in bull/bear markets: weekly trend filter avoids counter-trend trades

name = "4h_1d_1w_donchian_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d volume confirmation
    volume_1d = df_1d['volume'].values
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (2.0 * avg_volume_20_1d)
    
    # Pre-compute 1w EMA50 and EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    # Weekly trend: 1 = bullish (EMA50 > EMA200), -1 = bearish (EMA50 < EMA200), 0 = neutral/chop
    weekly_trend_raw = np.where(ema_50_1w > ema_200_1w, 1, np.where(ema_50_1w < ema_200_1w, -1, 0))
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_raw)
    
    # Pre-compute 4h Donchian channels (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h volume (for entry confirmation)
    volume_4h = prices['volume'].values
    
    # Pre-compute 4h ATR(14) for stoploss
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.roll(close_4h, 1))
    tr3_4h = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h[0] = tr1_4h[0]
    atr_14 = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_spike_1d[i]) or np.isnan(weekly_trend_aligned[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price < Donchian low OR stoploss hit
            if close_4h[i] < donchian_low[i] or close_4h[i] < entry_price - 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Donchian high OR stoploss hit
            if close_4h[i] > donchian_high[i] or close_4h[i] > entry_price + 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume spike and weekly trend filter
            if vol_spike_1d[i] and weekly_trend_aligned[i] != 0:
                # Long: price > Donchian high(20) AND weekly trend bullish
                if close_4h[i] > donchian_high[i] and weekly_trend_aligned[i] == 1:
                    position = 1
                    entry_price = close_4h[i]
                    signals[i] = 0.25
                # Short: price < Donchian low(20) AND weekly trend bearish
                elif close_4h[i] < donchian_low[i] and weekly_trend_aligned[i] == -1:
                    position = -1
                    entry_price = close_4h[i]
                    signals[i] = -0.25
    
    return signals