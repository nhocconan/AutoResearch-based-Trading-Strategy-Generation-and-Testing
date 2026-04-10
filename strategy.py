#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR expansion filter and 1w EMA trend filter
# - Entry: Long when price breaks above 4h Donchian H20 + 1d ATR(14) > 1.5x 20-period ATR MA + 1w EMA50 > EMA200 (bullish regime)
#          Short when price breaks below 4h Donchian L20 + 1d ATR(14) > 1.5x 20-period ATR MA + 1w EMA50 < EMA200 (bearish regime)
# - Exit: Close-based reversal - exit long when price < 4h Donchian L20, exit short when price > 4h Donchian H20
# - Stoploss: ATR-based - exit when price moves against position by 2.0 * ATR(14)
# - Position sizing: 0.25 (discrete level)
# - Uses volatility expansion (ATR > MA) to confirm genuine breakouts, EMA crossover for regime filter
# - Target: 75-150 total trades over 4 years (19-38/year) to stay well below HARD MAX: 400 total
# - ATR expansion filter reduces false breakouts during low volatility, EMA regime ensures trading with higher timeframe trend

name = "4h_1d_1w_donchian_atr_ema_v1"
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
    
    # Pre-compute 4h OHLC
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Pre-compute 1d data for ATR
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pre-compute 1w data for EMA
    close_1w = df_1w['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_h20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_l20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ATR (14-period)
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr1_1d[0] = 0
    tr2_1d[0] = 0
    tr3_1d[0] = 0
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR moving average (20-period) for expansion filter
    atr_ma_20_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w EMA50 and EMA200 for regime filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_bullish = ema50_1w > ema200_1w  # True for bullish regime
    ema_bearish = ema50_1w < ema200_1w  # True for bearish regime
    
    # Calculate 4h ATR (14-period) for stoploss
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.roll(close_4h, 1))
    tr3_4h = np.abs(low_4h - np.roll(close_4h, 1))
    tr1_4h[0] = 0
    tr2_4h[0] = 0
    tr3_4h[0] = 0
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    # Align all HTF data to 4h timeframe
    donchian_h20_aligned = align_htf_to_ltf(prices, prices, donchian_h20)  # 4h data already aligned
    donchian_l20_aligned = align_htf_to_ltf(prices, prices, donchian_l20)  # 4h data already aligned
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20_1d)
    ema_bullish_aligned = align_htf_to_ltf(prices, df_1w, ema_bullish.astype(float))
    ema_bearish_aligned = align_htf_to_ltf(prices, df_1w, ema_bearish.astype(float))
    atr_4h_aligned = align_htf_to_ltf(prices, prices, atr_4h)  # 4h data already aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(donchian_h20_aligned[i]) or np.isnan(donchian_l20_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(atr_ma_aligned[i]) or 
            np.isnan(ema_bullish_aligned[i]) or np.isnan(ema_bearish_aligned[i]) or 
            np.isnan(atr_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 4h close
        close_price = close_4h[i]
        
        # ATR expansion filter: current ATR > 1.5x 20-period ATR MA
        atr_expansion = atr_1d_aligned[i] > 1.5 * atr_ma_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian H20 + ATR expansion + bullish regime
            if (close_price > donchian_h20_aligned[i] and 
                atr_expansion and 
                ema_bullish_aligned[i] > 0.5):
                position = 1
                entry_price = close_price
                signals[i] = 0.25
            # Short entry: price breaks below Donchian L20 + ATR expansion + bearish regime
            elif (close_price < donchian_l20_aligned[i] and 
                  atr_expansion and 
                  ema_bearish_aligned[i] > 0.5):
                position = -1
                entry_price = close_price
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or stoploss
            # Calculate stoploss level
            if position == 1:  # Long position
                stop_loss = entry_price - 2.0 * atr_4h_aligned[i]
                # Exit conditions: price < Donchian L20 OR stoploss hit
                if close_price < donchian_l20_aligned[i] or close_price <= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short position
                stop_loss = entry_price + 2.0 * atr_4h_aligned[i]
                # Exit conditions: price > Donchian H20 OR stoploss hit
                if close_price > donchian_h20_aligned[i] or close_price >= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals