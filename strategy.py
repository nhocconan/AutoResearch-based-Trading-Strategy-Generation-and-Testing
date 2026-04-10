#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 12h volume spike and 1d trend filter
# - Long when Williams %R(14) crosses above -80 (oversold) AND 12h volume > 1.5x 20-period volume SMA AND 1d close > 1d EMA50
# - Short when Williams %R(14) crosses below -20 (overbought) AND 12h volume > 1.5x 20-period volume SMA AND 1d close < 1d EMA50
# - Exit: Williams %R returns to -50 (mean reversion) or volume drops below average
# - Williams %R identifies overextended moves ready for reversal
# - Volume confirmation ensures reversals have participation
# - 1d EMA50 filter avoids counter-trend trades in strong trends
# - Target: 12-30 trades/year to minimize fee drag while capturing high-probability reversals

name = "6h_12h_1d_williamsr_reversal_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 12h data ONCE before loop for volume confirmation (MTF rule compliance)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return signals
    
    # Load 1d data ONCE before loop for trend filter (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute Williams %R for 6h data (14-period)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Pre-compute volume SMA for 12h data (20-period)
    volume_12h = df_12h['volume'].values
    volume_sma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_sma_20_12h)
    
    # Pre-compute EMA50 for 1d close (trend filter)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 1d close aligned for trend comparison
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    for i in range(14, n):  # Start after 14-bar warmup for Williams %R
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(williams_r[i-1]) or 
            np.isnan(volume_sma_20_12h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(close_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 12h volume > 1.5x 20-period volume SMA
        vol_12h_current = df_12h['volume'].values
        vol_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_12h_current)
        vol_confirm = vol_12h_aligned[i] > 1.5 * volume_sma_20_12h_aligned[i]
        
        # Trend filter: 1d close vs EMA50
        uptrend = close_1d_aligned[i] > ema_50_1d_aligned[i]
        downtrend = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        # Williams %R reversal signals
        williams_r_long = williams_r[i] > -80 and williams_r[i-1] <= -80  # Cross above -80 (oversold)
        williams_r_short = williams_r[i] < -20 and williams_r[i-1] >= -20  # Cross below -20 (overbought)
        
        # Exit condition: Williams %R returns to -50 (mean reversion)
        exit_long = williams_r[i] >= -50
        exit_short = williams_r[i] <= -50
        
        # Trading logic
        if vol_confirm:
            # Long: Williams %R bullish reversal in uptrend
            if williams_r_long and uptrend:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            # Short: Williams %R bearish reversal in downtrend
            elif williams_r_short and downtrend:
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