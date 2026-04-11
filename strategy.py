#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_keltner_breakout_v1"
timeframe = "1d"
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return signals
    
    # Calculate weekly Keltner Channel (20-period EMA, 2.0 ATR)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # ATR(20) for weekly
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # EMA(20) for weekly
    ema_20 = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channels
    upper_keltner = ema_20 + 2.0 * atr_20
    lower_keltner = ema_20 - 2.0 * atr_20
    
    # Align weekly Keltner channels to daily
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1w, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1w, lower_keltner)
    
    # Daily volume confirmation: volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_keltner_aligned[i]) or np.isnan(lower_keltner_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_current > 1.5 * vol_ma_20[i]
        
        # Breakout conditions using weekly Keltner Channels
        breakout_up = price_close > upper_keltner_aligned[i]  # Break above upper Keltner
        breakout_down = price_close < lower_keltner_aligned[i]  # Break below lower Keltner
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Break above upper Keltner with volume confirmation
        if breakout_up and vol_confirm:
            enter_long = True
        
        # Short: Break below lower Keltner with volume confirmation
        if breakout_down and vol_confirm:
            enter_short = True
        
        # Exit conditions: return to opposite Keltner level (mean reversion)
        exit_long = price_close < ema_20_aligned[i]  # Return to weekly EMA
        exit_short = price_close > ema_20_aligned[i]  # Return to weekly EMA
        
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

# Hypothesis: 1d Keltner breakout strategy using weekly Keltner Channels (EMA20 ± 2*ATR20).
# Enters long when price breaks above weekly upper Keltner with volume > 1.5x 20-day average.
# Enters short when price breaks below weekly lower Keltner with volume > 1.5x 20-day average.
# Exits when price returns to weekly EMA(20) level.
# Weekly timeframe provides higher timeframe trend context, reducing false breakouts.
# Volume confirmation ensures breakouts are supported by participation.
# Target: 10-25 trades per year (40-100 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by capturing significant breakouts in either direction.