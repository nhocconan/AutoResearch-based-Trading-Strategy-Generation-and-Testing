#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d EMA filter and volume spike confirmation
# - Long: Williams %R(14) crosses above -80 from below + price > 1d EMA(50) + volume > 2.0x 20-period average
# - Short: Williams %R(14) crosses below -20 from above + price < 1d EMA(50) + volume > 2.0x 20-period average
# - Exit: Opposite Williams %R cross (%R > -20 for long exit, %R < -80 for short exit)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Williams %R identifies overbought/oversold conditions for mean reversion in ranging markets
# - 1d EMA(50) filter ensures trades align with higher timeframe trend
# - Volume spike confirmation filters weak signals and increases reliability
# - Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets

name = "6h_1d_williamsr_volume_v2"
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
    
    # Load 1d data ONCE before loop for EMA filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return signals
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 1d volume average for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute Williams %R on 6h timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_50_aligned[i]) or np.isnan(volume_sma_20_aligned[i]) or 
            np.isnan(williams_r[i]) or np.isnan(williams_r[i-1])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        williams_r_now = williams_r[i]
        williams_r_prev = williams_r[i-1]
        
        # 1d EMA trend filter
        price_above_ema = close_price > ema_50_aligned[i]
        price_below_ema = close_price < ema_50_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = volume_current > 2.0 * volume_sma_20_aligned[i]
        
        # Williams %R conditions
        wr_oversold = williams_r_now < -80
        wr_overbought = williams_r_now > -20
        wr_cross_up_oversold = (williams_r_prev <= -80) and (williams_r_now > -80)
        wr_cross_down_overbought = (williams_r_prev >= -20) and (williams_r_now < -20)
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Williams %R crosses above -80 from below (exiting oversold) + uptrend filter + volume spike
        if wr_cross_up_oversold and price_above_ema and vol_confirm:
            enter_long = True
        
        # Short: Williams %R crosses below -20 from above (entering overbought) + downtrend filter + volume spike
        if wr_cross_down_overbought and price_below_ema and vol_confirm:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long when Williams %R crosses above -20 (overbought)
            exit_long = wr_cross_down_overbought
        elif position == -1:
            # Exit short when Williams %R crosses below -80 (oversold)
            exit_short = wr_cross_up_oversold
        
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