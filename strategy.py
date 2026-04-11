#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R reversal with 1d trend filter and volume confirmation
# - Long: Williams %R(14) crosses above -80 (oversold) + price > 1d EMA50 + volume > 1.2x 20-period average
# - Short: Williams %R(14) crosses below -20 (overbought) + price < 1d EMA50 + volume > 1.2x 20-period average
# - Exit: Opposite Williams %R signal or ATR trailing stop (2.0 ATR from extreme)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 19-50 trades/year (75-200 total over 4 years) to stay within fee drag limits
# - Williams %R captures mean reversals in ranging markets and early trend exhaustion
# - 1d EMA50 filter ensures trades align with higher timeframe trend
# - Volume confirmation filters out low-momentum reversals
# - ATR stoploss manages risk during volatile periods

name = "4h_1d_williamsr_reversal_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    # Load 1d data ONCE before loop for trend filter and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 1d volume confirmation (20-period average)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute Williams %R on 4h timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # Avoid division by zero
    
    # Pre-compute ATR for stoploss (4h timeframe)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_sma_20_aligned[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Williams %R levels
        wr_current = williams_r[i]
        wr_previous = williams_r[i-1]
        
        # Volume confirmation: current volume > 1.2x 20-period average
        vol_confirm = volume_current > 1.2 * volume_sma_20_aligned[i]
        
        # Trend filter from 1d EMA50
        price_above_1d_ema = close_price > ema_50_1d_aligned[i]
        price_below_1d_ema = close_price < ema_50_1d_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long reversal: Williams %R crosses above -80 (oversold) with trend and volume confirmation
        if wr_previous <= -80 and wr_current > -80 and price_above_1d_ema and vol_confirm:
            enter_long = True
        
        # Short reversal: Williams %R crosses below -20 (overbought) with trend and volume confirmation
        if wr_previous >= -20 and wr_current < -20 and price_below_1d_ema and vol_confirm:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Williams %R crosses below -50 or hits ATR stoploss
            exit_long = (wr_current < -50) or (close_price <= long_stop)
        elif position == -1:
            # Exit short if Williams %R crosses above -50 or hits ATR stoploss
            exit_short = (wr_current > -50) or (close_price >= short_stop)
        
        # Update stoploss levels when entering a position
        if enter_long:
            entry_price = close_price
            long_stop = entry_price - 2.0 * atr_14[i]
        elif enter_short:
            entry_price = close_price
            short_stop = entry_price + 2.0 * atr_14[i]
        
        # Update trailing stoploss for existing positions
        if position == 1:
            # Trail long stop upward: max of current stop and (high - 2*ATR)
            long_stop = max(long_stop, high[i] - 2.0 * atr_14[i])
        elif position == -1:
            # Trail short stop downward: min of current stop and (low + 2*ATR)
            short_stop = min(short_stop, low[i] + 2.0 * atr_14[i])
        
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