# 1H STRATEGY: 4H/1D REGIME ADAPTIVE WITH VOLUME CONFIRMATION
# Strategy adapts to market regime using 1d EMA crossover and 4h Donchian breakouts
# Entry only during 08-20 UTC session with volume confirmation
# Position size: 0.20 to control drawdown and reduce fee churn
# Target: 60-150 total trades over 4 years (15-37/year)

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 1d data for regime detection (EMA crossover)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Load 4h data for entry signals (Donchian breakout)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 1d regime: EMA21 > EMA50 = bullish regime, EMA21 < EMA50 = bearish regime
    close_1d = df_1d['close'].values
    ema21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Volume confirmation using 4h volume
    vol_4h = df_4h['volume'].values
    vol_ma_20_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(ema21_1d_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma_20_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        price_close = prices['close'].iloc[i]
        vol_current = align_htf_to_ltf(prices, df_4h, vol_4h)[i]
        
        # Regime: bullish if EMA21 > EMA50, bearish if EMA21 < EMA50
        bullish_regime = ema21_1d_aligned[i] > ema50_1d_aligned[i]
        bearish_regime = ema21_1d_aligned[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = vol_current > 1.5 * vol_ma_20_4h_aligned[i]
        
        if position == 0:
            # Enter long: bullish regime + price breaks above Donchian high + volume
            if (bullish_regime and 
                price_close > donchian_high_aligned[i] and 
                volume_confirm):
                signals[i] = 0.20
                position = 1
            # Enter short: bearish regime + price breaks below Donchian low + volume
            elif (bearish_regime and 
                  price_close < donchian_low_aligned[i] and 
                  volume_confirm):
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Donchian low OR regime turns bearish
                if (price_close < donchian_low_aligned[i]) or (not bullish_regime):
                    exit_signal = True
            elif position == -1:
                # Exit short: price breaks above Donchian high OR regime turns bullish
                if (price_close > donchian_high_aligned[i]) or (not bearish_regime):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_RegimeAdaptive_Donchian_Volume"
timeframe = "1h"
leverage = 1.0