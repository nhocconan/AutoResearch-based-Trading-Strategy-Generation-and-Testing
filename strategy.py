#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_keltner_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Calculate daily ATR for Keltner channels
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift())
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift())
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate daily EMA for Keltner middle line
    ema_20 = pd.Series(df_1d['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Keltner channels: Upper = EMA + 2*ATR, Lower = EMA - 2*ATR
    keltner_upper = ema_20 + 2.0 * atr
    keltner_lower = ema_20 - 2.0 * atr
    
    # Align Keltner channels to 12h timeframe
    keltner_upper_aligned = align_htf_to_ltf(prices, df_1d, keltner_upper)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_1d, keltner_lower)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    
    # Volume confirmation: 20-period average on daily volume
    volume_sma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20)
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(keltner_upper_aligned[i]) or np.isnan(keltner_lower_aligned[i]) or
            np.isnan(ema_20_aligned[i]) or np.isnan(volume_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.3x 20-day average
        vol_confirm = volume_current > 1.3 * volume_sma_20_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Price breaks above Keltner Upper + volume confirmation
        if price_close > keltner_upper_aligned[i] and vol_confirm:
            enter_long = True
        
        # Short: Price breaks below Keltner Lower + volume confirmation
        if price_close < keltner_lower_aligned[i] and vol_confirm:
            enter_short = True
        
        # Exit conditions: price returns to the 20-day EMA (middle line)
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

# Hypothesis: Keltner channel breakout on daily timeframe with volume confirmation.
# Uses Keltner(20,2) channels: breakouts above upper channel signal strength,
# breakdowns below lower channel signal weakness. Volume confirmation ensures
# institutional participation. Works in both bull (breakouts) and bear (breakdowns).
# Exit when price returns to the 20-day EMA (mean reversion to the mean).
# Position size 0.25 balances risk and return. Target: 20-40 trades/year to minimize fee drag.