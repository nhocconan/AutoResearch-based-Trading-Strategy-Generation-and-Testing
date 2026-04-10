#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume confirmation and ATR-based position sizing
# - Long when price breaks above 20-period Donchian high AND 1w volume > 1.5x 20-period volume SMA
# - Short when price breaks below 20-period Donchian low AND 1w volume > 1.5x 20-period volume SMA
# - Exit: price retreats to midpoint of Donchian channel or ATR-based stoploss (2x ATR)
# - Position sizing: 0.30 discrete level to balance return and drawdown
# - Target: 20-50 trades/year on 1d timeframe to stay within fee drag limits
# - Uses 1w timeframe for volume confirmation and trend strength to avoid false breakouts

name = "1d_donchian_20_1w_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # Calculate 1d Donchian channel (20-period)
    high_rolling_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high = high_rolling_max
    donchian_low = low_rolling_min
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate 1w volume SMA for confirmation
    volume_1w = df_1w['volume'].values
    volume_sma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_sma_20_1w)
    
    # Calculate ATR(14) for stoploss
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(40, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_sma_20_1w_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1w volume > 1.5x 20-period volume SMA
        vol_confirm = volume_1w[i // 7] > 1.5 * volume_sma_20_1w_aligned[i] if i // 7 < len(volume_1w) else False
        
        # Breakout signals
        breakout_up = close[i] > donchian_high[i-1]  # Break above previous Donchian high
        breakout_down = close[i] < donchian_low[i-1]  # Break below previous Donchian low
        
        # Exit conditions: price retreats to midpoint OR ATR-based stoploss
        if position == 1:  # Long position
            exit_long = (close[i] < donchian_mid[i]) or (close[i] < close[i-1] - 2.0 * atr[i])
        elif position == -1:  # Short position
            exit_long = (close[i] > donchian_mid[i]) or (close[i] > close[i-1] + 2.0 * atr[i])
        else:
            exit_long = False
        
        if position == 0:  # Flat - look for entry
            if breakout_up and vol_confirm:
                position = 1
                signals[i] = 0.30
            elif breakout_down and vol_confirm:
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
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
    
    return signals