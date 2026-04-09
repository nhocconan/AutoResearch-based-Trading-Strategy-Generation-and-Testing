#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h/1d HTF for direction and 1h for timing
# - 4h Donchian breakout (20-period) for primary trend direction
# - 1d EMA200 filter to avoid counter-trend trades in strong regimes
# - 1h RSI(14) pullback to 50 for entry timing in direction of HTF trend
# - Volume confirmation: 1h volume > 1.5x 20-period average
# - Fixed position size 0.20 to control drawdown
# - Session filter: 08-20 UTC to avoid low-liquidity hours
# - Target: 15-30 trades/year (60-120 total over 4 years)

name = "1h_4h_1d_donchian_rsi_pullback_v1"
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
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian levels to 1h timeframe
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_20)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA200 for regime filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Pre-compute 1h indicators
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if outside session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(rsi_values[i]) or
            np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Fixed position size
        position_size = 0.20
        
        if position == 1:  # Long position
            # Exit when price closes below 4h Donchian low (trend change)
            if close[i] < donchian_low_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit when price closes above 4h Donchian high (trend change)
            if close[i] > donchian_high_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Entry conditions with volume confirmation and regime filter
            if volume_confirmed:
                # Regime filter: price relative to 1d EMA200
                above_ema200 = close[i] > ema_200_1d_aligned[i]
                below_ema200 = close[i] < ema_200_1d_aligned[i]
                
                # RSI pullback to 50 for entry timing
                rsi_bullish = 45 <= rsi_values[i] <= 55
                rsi_bearish = 45 <= rsi_values[i] <= 55
                
                # Long: break above 4h Donchian high in uptrend regime with RSI pullback
                if above_ema200 and close[i] > donchian_high_20_aligned[i] and rsi_bullish:
                    position = 1
                    signals[i] = position_size
                # Short: break below 4h Donchian low in downtrend regime with RSI pullback
                elif below_ema200 and close[i] < donchian_low_20_aligned[i] and rsi_bearish:
                    position = -1
                    signals[i] = -position_size
    
    return signals