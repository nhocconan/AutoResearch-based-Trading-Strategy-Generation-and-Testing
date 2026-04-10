#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d volume spike and ADX(14) regime filter
# - Long when Williams %R crosses above -80 (oversold) + 1d volume > 2.0x 20-period volume SMA + ADX < 25 (range/low trend)
# - Short when Williams %R crosses below -20 (overbought) + 1d volume > 2.0x 20-period volume SMA + ADX < 25
# - Exit: Williams %R returns to -50 level (mean reversion completion)
# - Position sizing: 0.25 discrete level
# - Williams %R identifies exhaustion points, volume confirms participation, ADX filter avoids strong trends where mean reversion fails
# - Works in bull/bear: mean reversion effective in ranging markets, ADX filter prevents trading against strong momentum
# - 12h timeframe targets 20-40 trades/year with strict entry conditions to minimize fee drag

name = "12h_1d_williamsr_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 12h Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 12h ADX(14) for regime filter (avoid strong trends)
    # True Range
    tr1 = np.maximum(high - low, 
                     np.maximum(np.abs(high - np.roll(close, 1)), 
                                np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]
    # Plus Directional Movement
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    plus_dm[0] = 0
    # Minus Directional Movement
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    minus_dm[0] = 0
    # Smoothed values
    atr = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    # Handle division by zero and invalid values
    plus_di = np.where(atr == 0, 0, plus_di)
    minus_di = np.where(atr == 0, 0, minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = np.where(np.isnan(adx) | np.isinf(adx), 0, adx)
    
    # Calculate 1d volume SMA(20) for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(williams_r[i-1]) or np.isnan(adx[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2.0x 20-period SMA (volume spike)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, df_1d['volume'].values)
        vol_confirm = vol_1d_current[i] > 2.0 * volume_sma_20_1d_aligned[i]
        
        # Regime filter: ADX < 25 indicates ranging/low trend market (favorable for mean reversion)
        ranging_market = adx[i] < 25
        
        # Williams %R mean reversion signals
        wr_now = williams_r[i]
        wr_prev = williams_r[i-1]
        wr_cross_above_80 = (wr_prev <= -80) and (wr_now > -80)  # Oversold bounce
        wr_cross_below_20 = (wr_prev >= -20) and (wr_now < -20)  # Overbought rejection
        
        # Entry conditions: Williams %R mean reversion with volume and regime confirmation
        long_entry = wr_cross_above_80 and vol_confirm and ranging_market
        short_entry = wr_cross_below_20 and vol_confirm and ranging_market
        
        # Exit conditions: Williams %R returns to -50 level (mean reversion completion)
        long_exit = wr_now >= -50  # Exit long when WR crosses above -50
        short_exit = wr_now <= -50  # Exit short when WR crosses below -50
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if long_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if short_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals