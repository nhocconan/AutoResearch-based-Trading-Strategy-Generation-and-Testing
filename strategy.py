#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and ATR-based stoploss
# - Long when price breaks above Camarilla H3 level AND 1d volume > 1.3x 20-period volume SMA
# - Short when price breaks below Camarilla L3 level AND 1d volume > 1.3x 20-period volume SMA
# - Exit: ATR-based trailing stop (2.5x ATR from extreme) or Camarilla midpoint (H3+L3)/2 reversion
# - Position sizing: 0.25 discrete level to balance return and drawdown
# - Target: 12-30 trades/year on 12h timeframe to stay within fee drag limits
# - Uses Camarilla pivots for structure, volume for confirmation, ATR for risk management

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
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    # H4 = RANGE * 1.1/2 + CLOSE
    # H3 = RANGE * 1.1/4 + CLOSE
    # H2 = RANGE * 1.1/6 + CLOSE
    # H1 = RANGE * 1.1/12 + CLOSE
    # L1 = CLOSE - RANGE * 1.1/12
    # L2 = CLOSE - RANGE * 1.1/6
    # L3 = CLOSE - RANGE * 1.1/4
    # L4 = CLOSE - RANGE * 1.1/2
    # where RANGE = HIGH - LOW
    
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    prev_range = prev_high - prev_low
    
    H3 = prev_close + prev_range * 1.1 / 4
    L3 = prev_close - prev_range * 1.1 / 4
    camarilla_mid = (H3 + L3) / 2.0  # Midpoint for exit
    
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
    
    for i in range(1, n):  # Start from 1 to have previous day data
        # Skip if any required data is invalid
        if (np.isnan(H3[i-1]) if i-1 < len(H3) else True or
            np.isnan(L3[i-1]) if i-1 < len(L3) else True or
            np.isnan(atr[i]) or np.isnan(volume_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.3x 20-period volume SMA
        # Get 1d volume for current 12h bar (each 1d bar = 2 12h bars)
        idx_1d = i // 2
        if idx_1d < len(volume_1d):
            vol_confirm = volume_1d[idx_1d] > 1.3 * volume_sma_20_1d_aligned[i]
        else:
            vol_confirm = False
        
        # Camarilla breakout signals (using previous day's levels)
        breakout_up = close[i] > H3[i-1]  # Break above H3
        breakout_down = close[i] < L3[i-1]  # Break below L3
        
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
        exit_long = close[i] < camarilla_mid[i-1]
        exit_short = close[i] > camarilla_mid[i-1]
        
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