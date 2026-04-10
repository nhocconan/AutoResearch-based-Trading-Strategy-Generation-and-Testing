#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + Weekly Trend Filter
# - Elder Ray: Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# - Long when Bull Power > 0 AND weekly EMA(34) rising AND volume > 1.2x 20-period volume SMA
# - Short when Bear Power > 0 AND weekly EMA(34) falling AND volume > 1.2x 20-period volume SMA
# - Exit: Opposite Elder Ray signal or ATR trailing stop (2.0x ATR)
# - Uses weekly EMA for major trend filter (avoids counter-trend trades in strong trends)
# - Elder Ray measures bull/bear strength relative to EMA(13)
# - Session filter: 08-20 UTC to avoid low-volume Asian session noise
# - Position sizing: 0.25 discrete level to control drawdown and minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag while maintaining statistical significance
# - Works in both bull and bear markets: weekly trend filter ensures we trade with major trend,
#   Elder Ray captures momentum within that trend

name = "6h_1w_elder_ray_v1"
timeframe = "6h"
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
    open_time = prices['open_time'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load weekly data ONCE before loop (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return signals
    
    # Calculate weekly EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_rising = ema_34_1w > np.roll(ema_34_1w, 1)
    ema_34_1w_falling = ema_34_1w < np.roll(ema_34_1w, 1)
    # Handle first value
    ema_34_1w_rising[0] = False
    ema_34_1w_falling[0] = False
    # Align to 6h timeframe with proper delay (completed weekly bar only)
    ema_34_1w_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w_rising)
    ema_34_1w_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w_falling)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate EMA(13) for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA(13)
    bear_power = ema_13 - low   # Bear Power = EMA(13) - Low
    
    # Calculate 20-period volume SMA for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(20, n):  # Start from 20 to have sufficient lookback
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is invalid
        if (np.isnan(atr[i]) or np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_sma_20[i]) or
            np.isnan(ema_34_1w_rising_aligned[i]) or np.isnan(ema_34_1w_falling_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 6h volume > 1.2x 20-period volume SMA
        vol_confirm = volume[i] > 1.2 * volume_sma_20[i]
        
        # Elder Ray signals
        bull_strong = bull_power[i] > 0  # Bull Power positive
        bear_strong = bear_power[i] > 0  # Bear Power positive
        
        if position == 0:  # Flat - look for entry
            # Long: Bull Power > 0 AND weekly EMA rising AND volume confirmation
            if bull_strong and ema_34_1w_rising_aligned[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Short: Bear Power > 0 AND weekly EMA falling AND volume confirmation
            elif bear_strong and ema_34_1w_falling_aligned[i] and vol_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            # Exit conditions: Bear Power becomes positive (trend weakness) OR ATR trailing stop
            exit_condition = bear_strong or (close[i] < (high[i - 1] if i > 0 else close[i]) - 2.0 * atr[i])
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            # Exit conditions: Bull Power becomes positive (trend weakness) OR ATR trailing stop
            exit_condition = bull_strong or (close[i] > (low[i - 1] if i > 0 else close[i]) + 2.0 * atr[i])
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals