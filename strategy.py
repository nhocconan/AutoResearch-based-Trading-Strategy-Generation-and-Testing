#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d ATR-based volatility breakout with 1w RSI regime filter
# - Uses 1d HTF for ATR breakout: price breaks above/below ATR(14) multiplier from open
# - Uses 1w HTF for RSI regime: RSI(14) > 50 for bullish bias, < 50 for bearish bias
# - In bullish regime (weekly RSI > 50): look for long breakouts above open + k*ATR
# - In bearish regime (weekly RSI < 50): look for short breakdowns below open - k*ATR
# - Volume confirmation: current 12h volume > 1.5x 20-period average to filter low-quality breakouts
# - Fixed position size 0.25 to control drawdown and enable discrete sizing
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

name = "12h_1d_1w_atr_breakout_rsi_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d ATR(14) for volatility measurement
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1w RSI(14) for regime filter
    delta = pd.Series(close_1w).diff().values
    delta[0] = 0
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Align all HTF data to 12h timeframe (wait for completed HTF bar)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    rsi_14_aligned = align_htf_to_ltf(prices, df_1w, rsi_14)
    
    # Pre-compute volume confirmation (20-period average for 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_14_aligned[i]) or np.isnan(rsi_14_aligned[i]) or
            np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Regime filter: weekly RSI > 50 = bullish bias, < 50 = bearish bias
        bullish_regime = rsi_14_aligned[i] > 50
        bearish_regime = rsi_14_aligned[i] < 50
        
        # ATR breakout levels from current bar's open
        atr_mult = 1.5  # ATR multiplier for breakout threshold
        upper_break = open_price[i] + atr_mult * atr_14_aligned[i]
        lower_break = open_price[i] - atr_mult * atr_14_aligned[i]
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit conditions: close below open (mean reversion) or regime change
            if close[i] < open_price[i] or not bullish_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit conditions: close above open (mean reversion) or regime change
            if close[i] > open_price[i] or not bearish_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Entry logic based on regime and ATR breakout
            if volume_confirmed:
                if bullish_regime and high[i] > upper_break:
                    # Bullish regime + upward breakout: long
                    position = 1
                    signals[i] = position_size
                elif bearish_regime and low[i] < lower_break:
                    # Bearish regime + downward breakout: short
                    position = -1
                    signals[i] = -position_size
    
    return signals