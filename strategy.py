#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d volume spike and 1w trend filter
# - Long: Williams %R(14) < -80 (oversold), volume > 2.0x 20-period avg, 1w EMA(50) > EMA(200) (bullish trend)
# - Short: Williams %R(14) > -20 (overbought), volume > 2.0x 20-period avg, 1w EMA(50) < EMA(200) (bearish trend)
# - Exit: Williams %R returns to -50 level or ATR-based stop
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Williams %R works well for mean reversion in ranging markets, volume confirms conviction,
#   1w EMA crossover ensures we trade with the higher timeframe trend

name = "12h_1d_1w_williamsr_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return signals
    
    # Pre-compute 1w EMA(50) and EMA(200) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1w EMAs to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Load 1d data ONCE before loop for Williams %R and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Pre-compute 1d Williams %R(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align 1d Williams %R to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Pre-compute 1d volume confirmation (20-period average)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute ATR for stoploss (12h timeframe)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(volume_sma_20_aligned[i]) or 
            np.isnan(atr_14[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Williams %R
        wr = williams_r_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = volume_current > 2.0 * volume_sma_20_aligned[i]
        
        # Trend filter: 1w EMA(50) > EMA(200) for bullish, < for bearish
        ema_bullish = ema_50_aligned[i] > ema_200_aligned[i]
        ema_bearish = ema_50_aligned[i] < ema_200_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: oversold + volume confirmation + bullish trend
        if wr < -80 and vol_confirm and ema_bullish:
            enter_long = True
        
        # Short: overbought + volume confirmation + bearish trend
        if wr > -20 and vol_confirm and ema_bearish:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Williams %R returns to -50 or ATR-based stop
            exit_long = (wr >= -50) or (close_price <= entry_price - 2.0 * atr_14[i])
        elif position == -1:
            # Exit short if Williams %R returns to -50 or ATR-based stop
            exit_short = (wr <= -50) or (close_price >= entry_price + 2.0 * atr_14[i])
        
        # Track entry price for stoploss calculation
        if enter_long or enter_short:
            entry_price = close_price
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals