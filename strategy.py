#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter (HMA21) and volume confirmation
# - Long when price breaks above Donchian upper band AND 1w HMA21 trending up AND volume > 1.5x 20-period volume SMA
# - Short when price breaks below Donchian lower band AND 1w HMA21 trending down AND volume > 1.5x 20-period volume SMA
# - Exit: ATR trailing stop (2.5x ATR) or reversion to Donchian midpoint
# - Uses 1w for trend filter (avoids whipsaw in ranging markets) and 1d for precise entry/exit timing
# - Position sizing: 0.25 discrete level to control drawdown and minimize fee churn
# - Target: 7-25 trades/year (30-100 total over 4 years) to minimize fee drag while maintaining statistical significance
# - Donchian breakouts capture strong momentum moves that work in both bull and bear markets
# - 1w HMA21 filter ensures we only trade with the higher timeframe trend, reducing false breakouts
# - Volume confirmation adds conviction to breakout signals

name = "1d_1w_donchian_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1w data ONCE before loop (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return signals
    
    # Calculate 1w HMA21 for trend filter
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    close_1w = df_1w['close'].values
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    wma_half = wma(close_1w, half_len)
    wma_full = wma(close_1w, 21)
    wma_diff = 2 * wma_half - wma_full
    hma_21 = wma(wma_diff, sqrt_len)
    
    # Pad HMA array to match original length
    hma_21_padded = np.full(len(close_1w), np.nan)
    hma_21_padded[half_len:half_len + len(hma_21)] = hma_21
    
    # HMA is trending up when current > previous, down when current < previous
    hma_up = np.diff(hma_21_padded, prepend=hma_21_padded[0]) > 0
    hma_down = np.diff(hma_21_padded, prepend=hma_21_padded[0]) < 0
    
    # Align to 1d timeframe with proper delay (completed 1w bar only)
    hma_up_aligned = align_htf_to_ltf(prices, df_1w, hma_up)
    hma_down_aligned = align_htf_to_ltf(prices, df_1w, hma_down)
    
    # Calculate ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period volume SMA for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period)
    # Upper band = highest high over 20 periods
    # Lower band = lowest low over 20 periods
    # Middle band = (upper + lower) / 2
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Track highest high since entry for trailing stop (long)
    # Track lowest low since entry for trailing stop (short)
    highest_since_entry = np.full(n, np.nan)
    lowest_since_entry = np.full(n, np.nan)
    
    for i in range(20, n):  # Start from 20 to have sufficient lookback
        # Skip if any required data is invalid
        if (np.isnan(atr[i]) or np.isnan(volume_sma_20[i]) or
            np.isnan(hma_up_aligned[i]) or np.isnan(hma_down_aligned[i]) or
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(donchian_middle[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_upper[i-1]  # Break above upper band
        breakout_down = close[i] < donchian_lower[i-1]  # Break below lower band
        
        if position == 0:  # Flat - look for entry
            # Long: price breaks above upper band AND 1w HMA trending up AND volume confirmation
            if breakout_up and hma_up_aligned[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
                highest_since_entry[i] = high[i]  # Initialize trailing stop
            # Short: price breaks below lower band AND 1w HMA trending down AND volume confirmation
            elif breakout_down and hma_down_aligned[i] and vol_confirm:
                position = -1
                signals[i] = -0.25
                lowest_since_entry[i] = low[i]  # Initialize trailing stop
            else:
                signals[i] = 0.0
                # Carry forward NaN values for tracking
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
        elif position == 1:  # Long position - look for exit
            # Update highest high since entry
            highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            
            # ATR trailing stop: exit if price drops 2.5*ATR below highest high since entry
            trailing_stop = highest_since_entry[i] - 2.5 * atr[i]
            
            # Exit conditions: trailing stop hit OR reversion to midpoint
            exit_condition = (close[i] < trailing_stop) or (close[i] < donchian_middle[i])
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
                # Reset tracking arrays
                highest_since_entry[i] = np.nan
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = 0.25
                # Propagate tracking values
                highest_since_entry[i] = highest_since_entry[i]
                lowest_since_entry[i] = lowest_since_entry[i-1]
        else:  # position == -1 (Short position) - look for exit
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            
            # ATR trailing stop: exit if price rises 2.5*ATR above lowest low since entry
            trailing_stop = lowest_since_entry[i] + 2.5 * atr[i]
            
            # Exit conditions: trailing stop hit OR reversion to midpoint
            exit_condition = (close[i] > trailing_stop) or (close[i] > donchian_middle[i])
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
                # Reset tracking arrays
                highest_since_entry[i] = np.nan
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = -0.25
                # Propagate tracking values
                highest_since_entry[i] = highest_since_entry[i-1]
                lowest_since_entry[i] = lowest_since_entry[i]
    
    return signals