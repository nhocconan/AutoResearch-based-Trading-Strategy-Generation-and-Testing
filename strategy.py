#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h volume confirmation and 1d trend filter
# - Long when price breaks above H3 Camarilla pivot level with 4h volume spike and daily uptrend
# - Short when price breaks below L3 Camarilla pivot level with 4h volume spike and daily downtrend
# - Uses 1h timeframe targeting 60-150 trades over 4 years (15-37/year) to minimize fee drag
# - Daily close > EMA50 for uptrend filter, close < EMA50 for downtrend (avoid counter-trend trades)
# - Volume confirmation: current 4h volume > 2.0x 20-period average to filter weak breakouts
# - Discrete position sizing (0.20) to minimize fee churn and control drawdown
# - Session filter: only trade 08-20 UTC to avoid low-liquidity periods
# - ATR-based trailing stop: exit when price moves 2.0x ATR against position

name = "1h_4h_1d_camarilla_breakout_volume_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 20 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 4h volume confirmation
    volume_4h = df_4h['volume'].values
    avg_volume_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_spike_4h = volume_4h > (2.0 * avg_volume_20)
    vol_spike_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_spike_4h)
    
    # Pre-compute 1h ATR(14) for stoploss
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    
    tr1 = high_1h - low_1h
    tr2 = np.abs(high_1h - np.roll(close_1h, 1))
    tr3 = np.abs(low_1h - np.roll(close_1h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Pre-compute 1h Camarilla pivots (using previous day's OHLC)
    # We need to align daily OHLC to 1h bars
    # Get daily OHLC from 1d data
    daily_open = df_1d['open'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_H3 = daily_close + 1.1 * (daily_high - daily_low) / 6
    camarilla_L3 = daily_close - 1.1 * (daily_high - daily_low) / 6
    
    # Align daily Camarilla levels to 1h timeframe (use previous day's levels)
    H3_daily = camarilla_H3
    L3_daily = camarilla_L3
    H3_1h_aligned = align_htf_to_ltf(prices, df_1d, H3_daily)
    L3_1h_aligned = align_htf_to_ltf(prices, df_1d, L3_daily)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(H3_1h_aligned[i]) or 
            np.isnan(L3_1h_aligned[i]) or np.isnan(vol_spike_4h_aligned[i]) or 
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss or price breaks below L3 (trend reversal)
            if (prices['close'].iloc[i] < entry_price - 2.0 * atr_14[i] or 
                prices['close'].iloc[i] < L3_1h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss or price breaks above H3 (trend reversal)
            if (prices['close'].iloc[i] > entry_price + 2.0 * atr_14[i] or 
                prices['close'].iloc[i] > H3_1h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for Camarilla breakout with trend and volume filters
            if vol_spike_4h_aligned[i]:
                # Long signal: price breaks above H3 in daily uptrend
                if (prices['close'].iloc[i] > H3_1h_aligned[i] and 
                    prices['close'].iloc[i] > ema_50_1d_aligned[i]):
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = 0.20
                # Short signal: price breaks below L3 in daily downtrend
                elif (prices['close'].iloc[i] < L3_1h_aligned[i] and 
                      prices['close'].iloc[i] < ema_50_1d_aligned[i]):
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = -0.20
    
    return signals