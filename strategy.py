#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w HMA trend filter and volume confirmation
# - Long when price breaks above Donchian(20) high AND 1w HMA(21) is rising AND 1d volume > 1.3x 20-period volume SMA
# - Short when price breaks below Donchian(20) low AND 1w HMA(21) is falling AND 1d volume > 1.3x 20-period volume SMA
# - Exit: opposite Donchian breakout or volume drops below average
# - Uses 1d for Donchian and volume, 1w for HMA trend filter
# - Weekly HMA provides smoothed trend direction to avoid whipsaws
# - Volume confirmation ensures breakouts have conviction
# - Target: 15-25 trades/year to minimize fee drag while capturing sustained moves

name = "1d_1w_hma_volume_donchian_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1w data ONCE before loop for HMA trend (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # Calculate HMA(21) on 1w close data
    close_1w = df_1w['close'].values
    # HMA formula: WMA(2 * WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    wma_half = np.array([np.nan] * len(close_1w))
    wma_full = np.array([np.nan] * len(close_1w))
    
    if len(close_1w) >= half_len:
        wma_half[half_len-1:] = wma(close_1w, half_len)
    if len(close_1w) >= 21:
        wma_full[20:] = wma(close_1w, 21)
    
    # HMA = WMA(2*WMA_half - WMA_full), sqrt_len)
    hma_input = 2 * wma_half - wma_full
    hma_1w = np.array([np.nan] * len(close_1w))
    if len(hma_input) >= sqrt_len:
        hma_1w[sqrt_len-1:] = wma(hma_input, sqrt_len)
    
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Pre-compute Donchian channels for 1d data (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute volume SMA for 1d data (20-period)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(20, n):  # Start after 20-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_sma_20[i]) or np.isnan(hma_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.3x 20-period volume SMA
        vol_confirm = volume[i] > 1.3 * volume_sma_20[i]
        
        # Donchian breakout signals (using prior period's levels)
        breakout_long = close[i] > donchian_high[i-1]  # Break above prior period's high
        breakout_short = close[i] < donchian_low[i-1]  # Break below prior period's low
        
        # Weekly HMA trend: rising if current > previous, falling if current < previous
        hma_rising = hma_1w_aligned[i] > hma_1w_aligned[i-1] if i > 0 else False
        hma_falling = hma_1w_aligned[i] < hma_1w_aligned[i-1] if i > 0 else False
        
        # Exit conditions: opposite breakout or volume drops below average
        exit_long = close[i] < donchian_low[i-1] or volume[i] < volume_sma_20[i]
        exit_short = close[i] > donchian_high[i-1] or volume[i] < volume_sma_20[i]
        
        # Trading logic
        if vol_confirm:
            # Long: Donchian breakout above with rising weekly HMA
            if breakout_long and hma_rising:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            # Short: Donchian breakout below with falling weekly HMA
            elif breakout_short and hma_falling:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25
            else:
                # Check for exits
                if position == 1 and exit_long:
                    position = 0
                    signals[i] = 0.0
                elif position == -1 and exit_short:
                    position = 0
                    signals[i] = 0.0
                else:
                    # Maintain current position
                    signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        else:
            # No volume confirmation: exit any position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals