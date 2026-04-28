# [Experiment 103188] Hypothesis: 12h Donchian(15) breakout with 1w EMA40 trend filter and volume confirmation
# Targets 20-30 trades/year on 12h timeframe to avoid fee drift. Uses weekly trend for multi-timeframe alignment.
# Works in bull via breakouts, in bear via filtering out counter-trend noise.
# Volume and volatility filters prevent whipsaws in low-momentum regimes.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data once for HTF trend context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # Calculate 1w indicators
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # 1w Donchian channel (15) - tighter for fewer, higher-quality signals
    donchian_high = pd.Series(high_1w).rolling(window=15, min_periods=15).max().values
    donchian_low = pd.Series(low_1w).rolling(window=15, min_periods=15).min().values
    
    # 1w EMA40 - smooth trend filter
    ema_40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    
    # 1w ATR14 - volatility filter
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align HTF indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    ema_40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w)
    atr_14_aligned = align_htf_to_ltf(prices, df_1w, atr_14)
    
    # Volume ratio for confirmation (20-period MA on weekly)
    vol_ma_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume_1w / vol_ma_1w
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1w, vol_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_40_1w_aligned[i]) or np.isnan(atr_14_aligned[i]) or
            np.isnan(vol_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > donchian_high_aligned[i]
        breakout_down = close[i] < donchian_low_aligned[i]
        
        # Trend filter: price above/below 1w EMA40
        trend_up = close[i] > ema_40_1w_aligned[i]
        trend_down = close[i] < ema_40_1w_aligned[i]
        
        # Volatility filter: avoid extremely low volatility periods
        vol_filter = atr_14_aligned[i] > 0.006 * close[i]  # ATR > 0.6% of price
        
        # Volume confirmation: require above-average volume
        vol_confirm = vol_ratio_aligned[i] > 1.3
        
        # Entry conditions - optimized for 12h timeframe
        # Long: upward breakout + uptrend + vol filter + volume confirmation
        long_entry = breakout_up and trend_up and vol_filter and vol_confirm
        # Short: downward breakout + downtrend + vol filter + volume confirmation
        short_entry = breakout_down and trend_down and vol_filter and vol_confirm
        
        # Exit conditions: opposite breakout or trend reversal
        long_exit = breakout_down or not trend_up
        short_exit = breakout_up or not trend_down
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Donchian15_Breakout_1wEMA40_Volume_Filter"
timeframe = "12h"
leverage = 1.0