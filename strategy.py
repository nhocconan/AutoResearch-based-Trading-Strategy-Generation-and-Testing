#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Donchian channels provide clear breakout levels that work in all market regimes
# 12h EMA50 provides intermediate-term trend filter to reduce false breakouts
# Volume confirmation (>1.5x average) ensures breakout legitimacy
# ATR-based stoploss manages risk
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and fee drag

name = "4h_Donchian20_12hEMA50_Trend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period) from previous bar
    # Upper = max(high, lookback=20)
    # Lower = min(low, lookback=20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # ATR for stoploss (14-period)
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = 0
    tr3.iloc[0] = 0
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 20, 14)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_atr = atr[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on breakout with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish breakout: price above Donchian upper + above 12h EMA50
                if curr_close > donchian_upper[i] and curr_close > curr_ema_50_12h:
                    signals[i] = 0.30
                    position = 1
                    entry_price = curr_close
                # Bearish breakout: price below Donchian lower + below 12h EMA50
                elif curr_close < donchian_lower[i] and curr_close < curr_ema_50_12h:
                    signals[i] = -0.30
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit conditions: stoploss, trend reversal, or take profit
            # Stoploss: 2 * ATR below entry
            if curr_low <= entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Trend reversal: price below 12h EMA50
            elif curr_close < curr_ema_50_12h:
                signals[i] = 0.0
                position = 0
            # Take profit: price touches opposite Donchian band
            elif curr_high >= donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit conditions: stoploss, trend reversal, or take profit
            # Stoploss: 2 * ATR above entry
            if curr_high >= entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Trend reversal: price above 12h EMA50
            elif curr_close > curr_ema_50_12h:
                signals[i] = 0.0
                position = 0
            # Take profit: price touches opposite Donchian band
            elif curr_low <= donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals