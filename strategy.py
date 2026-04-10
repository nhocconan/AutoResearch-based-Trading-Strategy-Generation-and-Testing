#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume confirmation and ADX(14) > 25 trend filter
# - Long when price breaks above Donchian upper + 1w volume > 1.5x 20-period volume SMA + ADX > 25
# - Short when price breaks below Donchian lower + 1w volume > 1.5x 20-period volume SMA + ADX > 25
# - Exit: price returns to Donchian middle (mean reversion) or opposite breakout
# - Position sizing: 0.30 discrete level
# - Donchian captures institutional breakouts, volume confirms participation, ADX avoids false breakouts in chop
# - Works in bull/bear: breakouts occur in both regimes when volume confirms institutional interest
# - 1d timeframe targets 10-25 trades/year with strict entry conditions to minimize fee drag

name = "1d_1w_donchian_volume_adx_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 1d Donchian Channel (20-period)
    donch_period = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    for i in range(donch_period - 1, n):
        upper[i] = np.max(high[i-donch_period+1:i+1])
        lower[i] = np.min(low[i-donch_period+1:i+1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    # Calculate 1d ADX(14) for trend filter
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
    
    # Calculate 1w volume SMA(20) for confirmation
    volume_1w = df_1w['volume'].values
    volume_sma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_sma_20_1w)
    
    # Warmup period: need enough data for all indicators
    warmup = max(100, donch_period + 20)  # Ensure Donchian and volume SMA are valid
    
    for i in range(warmup, n):
        # Skip if any required data is invalid
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or 
            np.isnan(adx[i]) or np.isnan(volume_sma_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1w volume > 1.5x 20-period SMA (volume spike)
        vol_1w_current = align_htf_to_ltf(prices, df_1w, df_1w['volume'].values)
        vol_confirm = vol_1w_current[i] > 1.5 * volume_sma_20_1w_aligned[i]
        
        # Trend filter: ADX > 25 indicates strong trend (avoid choppy markets)
        trending_market = adx[i] > 25
        
        # Donchian breakout signals
        breakout_up = close[i] > upper[i-1]  # Break above previous period's upper band
        breakout_down = close[i] < lower[i-1]  # Break below previous period's lower band
        
        # Exit conditions: price returns to middle line (mean reversion)
        exit_long = close[i] < middle[i]
        exit_short = close[i] > middle[i]
        
        # Entry conditions: Donchian breakout with volume and trend confirmation
        long_entry = breakout_up and vol_confirm and trending_market
        short_entry = breakout_down and vol_confirm and trending_market
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.30
            elif short_entry:
                position = -1
                signals[i] = -0.30
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
    
    return signals