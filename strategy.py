#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout for trend direction and 1d RSI for mean reversion timing
# - Uses 4h HTF for Donchian channel (20-period): breakout above/below signals trend
# - Uses 1d HTF for RSI(14): extreme readings (<30 for long, >70 for short) provide entry timing
# - In bullish 4h trend (price > upper Donchian): look for long entries when 1d RSI < 30 (pullback)
# - In bearish 4h trend (price < lower Donchian): look for short entries when 1d RSI > 70 (bounce)
# - Volume confirmation: current 1h volume > 1.5x 20-period average to avoid low-volume false signals
# - Session filter: only trade 08-20 UTC to reduce noise
# - Fixed position size 0.20 to control drawdown and enable discrete levels
# - Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)

name = "1h_4h_1d_donchian_rsi_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    close_1d = df_1d['close'].values
    
    # Calculate 4h Donchian channel (20-period)
    period20_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_upper = period20_high
    donchian_lower = period20_low
    
    # Calculate 1d RSI(14)
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Align all HTF data to 1h timeframe (wait for completed HTF bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    # Pre-compute volume confirmation (20-period average for 1h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(rsi_aligned[i]) or np.isnan(vol_ma_20[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Fixed position size
        position_size = 0.20
        
        if position == 1:  # Long position
            # Exit conditions: trend changes or RSI normalizes
            if close[i] <= donchian_lower_aligned[i] or rsi_aligned[i] >= 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit conditions: trend changes or RSI normalizes
            if close[i] >= donchian_upper_aligned[i] or rsi_aligned[i] <= 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Entry logic based on 4h trend and 1d RSI extremes
            if volume_confirmed:
                # Bullish 4h trend: price above upper Donchian
                if close[i] > donchian_upper_aligned[i] and rsi_aligned[i] < 30:
                    position = 1
                    signals[i] = position_size
                # Bearish 4h trend: price below lower Donchian
                elif close[i] < donchian_lower_aligned[i] and rsi_aligned[i] > 70:
                    position = -1
                    signals[i] = -position_size
    
    return signals