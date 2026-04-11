#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_keltner_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return signals
    
    # Weekly high, low, close
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Keltner Channels (20-period EMA, 2x ATR)
    atr_period = 20
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    ema_20 = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    kc_upper = ema_20 + 2.0 * atr
    kc_lower = ema_20 - 2.0 * atr
    
    # Volume: 20-period SMA
    volume_sma_20 = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    
    # Align to daily timeframe
    kc_upper_aligned = align_htf_to_ltf(prices, df_1w, kc_upper)
    kc_lower_aligned = align_htf_to_ltf(prices, df_1w, kc_lower)
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1w, volume_sma_20)
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(kc_upper_aligned[i]) or np.isnan(kc_lower_aligned[i]) or
            np.isnan(volume_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period weekly average
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Entry conditions
        enter_long = price_close > kc_upper_aligned[i] and vol_confirm
        enter_short = price_close < kc_lower_aligned[i] and vol_confirm
        
        # Exit: reverse signal or volatility contraction
        exit_long = price_close < ema_20_aligned[i] if 'ema_20_aligned' in locals() else False
        exit_short = price_close > ema_20_aligned[i] if 'ema_20_aligned' in locals() else False
        
        # Align EMA for exit
        if i >= 20:
            ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
            exit_long = price_close < ema_20_aligned[i]
            exit_short = price_close > ema_20_aligned[i]
        
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

# Hypothesis: Keltner breakout on weekly timeframe with volume confirmation.
# Buys when daily price breaks above weekly upper Keltner band (EMA20 + 2*ATR) with volume surge.
# Sells when breaks below lower band with volume surge.
# Works in both bull (breakouts above upper band) and bear (breakdowns below lower band).
# Volume confirmation ensures institutional participation. Weekly timeframe reduces noise.
# Position size 0.25 balances risk and return. Target: 15-25 trades/year to minimize fee drag.