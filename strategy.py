#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d EMA trend filter and volume spike confirmation
# - Long: Williams %R(14) < -80 (oversold), price > 1d EMA(50) (uptrend bias), volume > 2.0x 20-period avg
# - Short: Williams %R(14) > -20 (overbought), price < 1d EMA(50) (downtrend bias), volume > 2.0x 20-period avg
# - Exit: Williams %R returns to -50 (mean reversion) or opposite extreme
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-30 trades/year (50-120 total over 4 years) to stay within fee drag limits
# - Williams %R is effective in ranging markets; EMA filter avoids counter-trend trades in strong trends

name = "6h_1d_williamsr_ema_volume_v1"
timeframe = "6h"
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
    
    # Load 1d data ONCE before loop for EMA trend filter (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return signals
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA(50) to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 6h Williams %R(14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Pre-compute 6h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(volume_sma_20[i]) or np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Williams %R levels
        wr = williams_r[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = volume_current > 2.0 * volume_sma_20[i]
        
        # Trend filter: price vs 1d EMA(50)
        price_above_ema = close_price > ema_50_aligned[i]
        price_below_ema = close_price < ema_50_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: oversold + uptrend bias + volume spike
        if wr < -80 and price_above_ema and vol_confirm:
            enter_long = True
        
        # Short: overbought + downtrend bias + volume spike
        if wr > -20 and price_below_ema and vol_confirm:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long when Williams %R returns to mean (-50) or goes overbought
            exit_long = (wr >= -50) or (wr > -20)
        elif position == -1:
            # Exit short when Williams %R returns to mean (-50) or goes oversold
            exit_short = (wr <= -50) or (wr < -80)
        
        # Track entry price for stoploss calculation (not used in exits but required for logic)
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