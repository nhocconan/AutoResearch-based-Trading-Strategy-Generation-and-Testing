#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_WeeklyTrend_VolumeSpike_v1
Hypothesis: On daily timeframe, Donchian(20) breakouts aligned with weekly EMA50 trend and volume spike (>1.5x 20-day average volume) capture sustained moves. Uses ATR-based stoploss (signal=0 when price retraces 1.5*ATR from extreme) and discrete sizing (0.25) to minimize fee churn. Designed for 30-80 trades over 4 years (7-20/year) with Sharpe > 0 in both bull and bear markets via trend filter and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for EMA trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1-week EMA50 for trend filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Donchian channels (20-period) on daily ===
    high = prices['high'].values
    low = prices['low'].values
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume spike filter (>1.5x 20-day average volume) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma_20)
    
    # === ATR (14) for stoploss ===
    tr1 = pd.Series(prices['high']).rolling(2).apply(lambda x: x[0] - x[1]).abs().values
    tr2 = pd.Series(prices['low']).rolling(2).apply(lambda x: x[0] - x[1]).abs().values
    tr3 = pd.Series(prices['close']).rolling(2).apply(lambda x: abs(x[0] - x[1])).values
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        ema_50 = ema_50_1w_aligned[i]
        upper_channel = donchian_high[i]
        lower_channel = donchian_low[i]
        vol_spike_now = vol_spike[i]
        atr_now = atr[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian + above weekly EMA50 + volume spike
            if price_high > upper_channel and price_close > ema_50 and vol_spike_now:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
                highest_since_entry = price_close
            # Short: price breaks below lower Donchian + below weekly EMA50 + volume spike
            elif price_low < lower_channel and price_close < ema_50 and vol_spike_now:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
                lowest_since_entry = price_close
        
        elif position != 0:
            # Update extremes
            if position == 1:
                highest_since_entry = max(highest_since_entry, price_high)
                # Exit: stoploss (1.5*ATR from high) OR re-entry into channel OR trend weakening
                if (price_low < highest_since_entry - 1.5 * atr_now or
                    price_close < lower_channel or
                    price_close < ema_50):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, price_low)
                # Exit: stoploss (1.5*ATR from low) OR re-entry into channel OR trend weakening
                if (price_high > lowest_since_entry + 1.5 * atr_now or
                    price_close > upper_channel or
                    price_close > ema_50):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_WeeklyTrend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0