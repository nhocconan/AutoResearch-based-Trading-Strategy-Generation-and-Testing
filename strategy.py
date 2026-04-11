#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R + Donchian breakout with volume confirmation
# - Williams %R(14): Overbought > -20, Oversold < -80
# - Donchian(20): Upper/lower 20-period channels
# - Long: Williams %R crosses above -80 (oversold bounce) AND price > Donchian Upper(20) AND volume > 1.5x 20-period average
# - Short: Williams %R crosses below -20 (overbought rejection) AND price < Donchian Lower(20) AND volume > 1.5x 20-period average
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Williams %R identifies reversal points in both bull and bear markets
# - Donchian breakout provides trend confirmation and avoids false reversals
# - Volume confirmation ensures institutional participation
# - Works in bull markets (buy oversold dips in uptrend) and bear markets (sell overbought rallies in downtrend)

name = "12h_williamsr_donchian_volume_v1"
timeframe = "12h"
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
    
    # Load 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d volume confirmation (20-period average)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute Williams %R on 12h timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).values
    
    # Pre-compute Donchian channels on 12h timeframe
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(volume_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        volume_current = volume[i]
        price_close = close[i]
        
        # Williams %R conditions
        wr_oversold = williams_r[i] < -80
        wr_overbought = williams_r[i] > -20
        wr_cross_above_oversold = williams_r[i] > -80 and williams_r[i-1] <= -80
        wr_cross_below_overbought = williams_r[i] < -20 and williams_r[i-1] >= -20
        
        # Donchian breakout conditions
        price_above_upper = price_close > donchian_upper[i]
        price_below_lower = price_close < donchian_lower[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Williams %R crosses above -80 (oversold bounce) + price > Donchian Upper + volume confirmation
        if wr_cross_above_oversold and price_above_upper and vol_confirm:
            enter_long = True
        
        # Short: Williams %R crosses below -20 (overbought rejection) + price < Donchian Lower + volume confirmation
        if wr_cross_below_overbought and price_below_lower and vol_confirm:
            enter_short = True
        
        # Exit conditions: reverse Williams %R cross or loss of channel breakout
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Williams %R crosses below -80 OR price < Donchian Lower
            exit_long = wr_cross_below_overbought or price_close < donchian_lower[i]
        elif position == -1:
            # Exit short if Williams %R crosses above -20 OR price > Donchian Upper
            exit_short = wr_cross_above_oversold or price_close > donchian_upper[i]
        
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