#!/usr/bin/env python3
"""
exp_7555_6d_hybrid_6h_1w
Hypothesis: 6-hour Williams %R (14) overbought/oversold with weekly trend filter and volume confirmation.
- In weekly uptrend (price > weekly EMA50): buy when W%R crosses above -50 from oversold (< -80)
- In weekly downtrend (price < weekly EMA50): sell when W%R crosses below -50 from overbought (> -20)
- Volume must be above average to confirm momentum
- Williams %R is effective in ranging and trending markets, providing early reversal signals
- Targets 75-150 trades over 4 years (19-38/year) with moderate frequency
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7555_6d_hybrid_6h_1w"
timeframe = "6h"
leverage = 1.0

# Parameters
WILLIAMS_R_PERIOD = 14
EMA_TREND = 50
VOLUME_MA_PERIOD = 20
WR_OVERBOUGHT = -20
WR_OVERSOLD = -80
WR_CROSS_LEVEL = -50
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_1w_50 = pd.Series(close_1w).ewm(span=EMA_TREND, adjust=False, min_periods=EMA_TREND).mean().values
    ema_1w_50_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_50)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=WILLIAMS_R_PERIOD, min_periods=WILLIAMS_R_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=WILLIAMS_R_PERIOD, min_periods=WILLIAMS_R_PERIOD).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close) / (highest_high - lowest_low)) * -100,
        -50  # neutral when no range
    )
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(WILLIAMS_R_PERIOD, EMA_TREND, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_1w_50_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine weekly trend
        weekly_uptrend = close[i] > ema_1w_50_aligned[i]   # price above weekly EMA50
        weekly_downtrend = close[i] < ema_1w_50_aligned[i]  # price below weekly EMA50
        
        # Volume confirmation
        volume_confirmed = volume[i] > volume_ma[i] if not np.isnan(volume_ma[i]) else False
        
        # Williams %R signals
        wr_current = williams_r[i]
        wr_previous = williams_r[i-1] if i-1 >= 0 else -50
        
        # Long signal: W%R crosses above -50 from oversold in uptrend
        long_signal = (
            weekly_uptrend and
            wr_previous < WR_OVERSOLD and
            wr_current > WR_CROSS_LEVEL and
            volume_confirmed
        )
        
        # Short signal: W%R crosses below -50 from overbought in downtrend
        short_signal = (
            weekly_downtrend and
            wr_previous > WR_OVERBOUGHT and
            wr_current < WR_CROSS_LEVEL and
            volume_confirmed
        )
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals