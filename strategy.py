#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme + 1d volume spike + 1w trend filter
# - Williams %R(14) < -80 (oversold) + 1d volume > 2.0x 20-period volume average + 1w close > 1w EMA20 → long
# - Williams %R(14) > -20 (overbought) + 1d volume > 2.0x 20-period volume average + 1w close < 1w EMA20 → short
# - Exit when Williams %R returns to -50 level or ATR stoploss triggered (adverse move > 2.5*ATR)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Works in bull/bear: Williams %R captures exhaustion; volume confirms participation; weekly trend filter avoids counter-trend trades
# - Target: 12-30 trades/year to stay within fee drag limits while capturing strong reversals

name = "6h_1d_1w_williamsr_vol_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load 6h data ONCE before loop for Williams %R and ATR (MTF rule compliance)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return signals
    
    # Load 1d data ONCE before loop for volume confirmation (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Load 1w data ONCE before loop for trend filter (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute 6h Williams %R(14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # Pre-compute 6h ATR(20) for stoploss
    tr1 = pd.Series(high).rolling(2).max() - pd.Series(low).rolling(2).min()
    tr2 = abs(pd.Series(high).shift(1) - pd.Series(close))
    tr3 = abs(pd.Series(low).shift(1) - pd.Series(close))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_20 = tr.rolling(window=20, min_periods=20).mean().values
    atr_20_aligned = align_htf_to_ltf(prices, df_6h, atr_20)
    
    # Pre-compute 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute 1w EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(atr_20_aligned[i]) or
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        
        # Volume confirmation: 1d volume > 2.0x 20-period volume average (strict threshold)
        volume_1d_current = df_1d['volume'].values
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d_current)
        vol_confirm = volume_1d_aligned[i] > 2.0 * volume_sma_20_1d_aligned[i]
        
        # Weekly trend filter: close > EMA20 for bullish, close < EMA20 for bearish
        weekly_close = df_1w['close'].values
        weekly_close_aligned = align_htf_to_ltf(prices, df_1w, weekly_close)
        weekly_bullish = weekly_close_aligned[i] > ema_20_1w_aligned[i]
        weekly_bearish = weekly_close_aligned[i] < ema_20_1w_aligned[i]
        
        # Williams %R extreme conditions
        williams_r_oversold = williams_r_aligned[i] < -80
        williams_r_overbought = williams_r_aligned[i] > -20
        williams_r_exit = abs(williams_r_aligned[i] + 50) < 5  # Near -50 level
        
        # Entry conditions
        enter_long = williams_r_oversold and vol_confirm and weekly_bullish
        enter_short = williams_r_overbought and vol_confirm and weekly_bearish
        
        # Exit conditions
        exit_long = (position == 1 and 
                    (williams_r_exit or  # Return to neutral level
                     close_price < entry_price - 2.5 * atr_20_aligned[i]))  # ATR stoploss
        exit_short = (position == -1 and 
                     (williams_r_exit or  # Return to neutral level
                      close_price > entry_price + 2.5 * atr_20_aligned[i]))  # ATR stoploss
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            entry_price = close_price
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            entry_price = close_price
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            entry_price = 0.0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            entry_price = 0.0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals