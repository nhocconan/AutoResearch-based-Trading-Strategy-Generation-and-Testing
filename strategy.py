# The provided code snippet was truncated...
# Ensure your response is a complete, standalone Python script.
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly trend filter (1w EMA50) and volume confirmation.
# Captures strong trending moves by buying/selling breakouts only when aligned with the weekly trend.
# Works in bull markets (buy breakouts above weekly EMA50) and bear markets (sell breakouts below weekly EMA50).
# Target: 15-25 trades/year by requiring weekly trend alignment, Donchian breakout, and volume surge.
# Entry: Long when price breaks above 6h Donchian high(20) with volume > 1.5x 20-period average and price > weekly EMA50.
#        Short when price breaks below 6h Donchian low(20) with volume > 1.5x 20-period average and price < weekly EMA50.
# Exit: Opposite Donchian touch (long exits at Donchian low, short exits at Donchian high) or trailing stop (2x ATR).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on weekly timeframe
    close_w = df_1w['close'].values
    ema_50_w = pd.Series(close_w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA to 6h (wait for weekly close)
    ema_50_w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_w)
    
    # Pre-calculate 6h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    
    # Donchian high: max of last 20 highs
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Donchian low: min of last 20 lows
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average volume on 6h
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for trailing stop (20-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - prices['close'].values[:-1])
    tr3 = np.abs(low[1:] - prices['close'].values[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session hours (08-20 UTC) - optional but can help avoid low-volume periods
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track for trailing stop
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_50_w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC (can be removed if too restrictive)
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        price_close = prices['close'].iloc[i]
        
        # Trend filter: price relative to weekly EMA50
        above_weekly_ema = price_close > ema_50_w_aligned[i]
        below_weekly_ema = price_close < ema_50_w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Enter long: break above Donchian high with volume and above weekly EMA
            if price_close > donchian_high[i] and volume_confirm and above_weekly_ema:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
            # Enter short: break below Donchian low with volume and below weekly EMA
            elif price_close < donchian_low[i] and volume_confirm and below_weekly_ema:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price touches Donchian low or trailing stop hit (2x ATR from entry)
                if price_close <= donchian_low[i]:
                    exit_signal = True
                elif price_close < entry_price - 2.0 * atr[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: price touches Donchian high or trailing stop hit (2x ATR from entry)
                if price_close >= donchian_high[i]:
                    exit_signal = True
                elif price_close > entry_price + 2.0 * atr[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Donchian20_1wEMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0