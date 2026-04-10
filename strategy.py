#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and ATR stoploss
# - Long when price breaks above H3 level AND 1d volume > 1.5x 20-period volume SMA
# - Short when price breaks below L3 level AND 1d volume > 1.5x 20-period volume SMA
# - Exit: ATR-based trailing stop (2.5x ATR from extreme) or Camarilla midpoint reversion
# - Uses Camarilla pivots from daily timeframe for structure, volume for confirmation, ATR for risk
# - Position sizing: 0.25 discrete level to limit drawdown in bear markets
# - Target: 12-30 trades/year on 12h timeframe to stay within fee drag limits

name = "12h_1d_camarilla_volume_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # Calculate ATR(14) for stoploss
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Track extreme prices for trailing stop
    long_extreme = np.full(n, np.nan)
    short_extreme = np.full(n, np.nan)
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(atr[i]) or np.isnan(volume_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels from previous 1d bar (need 2 prior 1d bars for H3/L3)
        # Get the index of the completed 1d bar for current 12h bar
        idx_1d_completed = (i // 2) - 1  # Each 1d bar = 2 12h bars, -1 for completed bar
        
        if idx_1d_completed < 1:
            signals[i] = 0.0
            continue
            
        # Get high, low, close from completed 1d bar (2 bars back)
        idx_1d_prev = idx_1d_completed - 1
        if idx_1d_prev < 0 or idx_1d_prev >= len(df_1d):
            signals[i] = 0.0
            continue
            
        h1 = df_1d['high'].iloc[idx_1d_prev]
        l1 = df_1d['low'].iloc[idx_1d_prev]
        c1 = df_1d['close'].iloc[idx_1d_prev]
        
        # Calculate Camarilla levels
        rang = h1 - l1
        if rang <= 0:
            signals[i] = 0.0
            continue
            
        # Camarilla levels
        h3 = c1 + (rang * 1.1 / 4)
        l3 = c1 - (rang * 1.1 / 4)
        h4 = c1 + (rang * 1.1 / 2)
        l4 = c1 - (rang * 1.1 / 2)
        cam_mid = (h3 + l3) / 2.0
        
        # Volume confirmation: 1d volume > 1.5x 20-period volume SMA
        idx_1d_vol = idx_1d_completed
        if idx_1d_vol < 0 or idx_1d_vol >= len(volume_1d):
            vol_confirm = False
        else:
            vol_confirm = volume_1d[idx_1d_vol] > 1.5 * volume_sma_20_1d_aligned[i]
        
        # Camarilla breakout signals
        breakout_up = close[i] > h3  # Break above H3
        breakout_down = close[i] < l3  # Break below L3
        
        # Update extremes for trailing stop
        if position == 1:  # Long position
            if np.isnan(long_extreme[i-1]):
                long_extreme[i] = close[i]
            else:
                long_extreme[i] = max(long_extreme[i-1], close[i])
        elif position == -1:  # Short position
            if np.isnan(short_extreme[i-1]):
                short_extreme[i] = close[i]
            else:
                short_extreme[i] = min(short_extreme[i-1], close[i])
        else:
            long_extreme[i] = np.nan
            short_extreme[i] = np.nan
        
        # ATR-based trailing stop conditions
        stop_long = False
        stop_short = False
        
        if position == 1 and not np.isnan(long_extreme[i]):
            stop_long = close[i] < long_extreme[i] - 2.5 * atr[i]
        elif position == -1 and not np.isnan(short_extreme[i]):
            stop_short = close[i] > short_extreme[i] + 2.5 * atr[i]
        
        # Camarilla midpoint reversion exit
        exit_long = close[i] < cam_mid
        exit_short = close[i] > cam_mid
        
        if position == 0:  # Flat - look for entry
            if breakout_up and vol_confirm:
                position = 1
                signals[i] = 0.25
            elif breakout_down and vol_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if stop_long or exit_long:
                position = 0
                signals[i] = 0.0
                long_extreme[i] = np.nan  # Reset extreme
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if stop_short or exit_short:
                position = 0
                signals[i] = 0.0
                short_extreme[i] = np.nan  # Reset extreme
            else:
                signals[i] = -0.25
    
    return signals